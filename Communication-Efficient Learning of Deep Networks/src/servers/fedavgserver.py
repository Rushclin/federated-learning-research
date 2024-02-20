import os
import json
import torch
import random
import logging
import numpy as np
import concurrent.futures

from importlib import import_module
from collections import ChainMap, defaultdict

from src import TqdmToLogger, MetricManager
from .serverbase import BaseServer

logger = logging.getLogger(__name__)


class FedavgServer(BaseServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model):
        super(FedavgServer, self).__init__()
        self.args = args
        self.writer = writer

        self.round = 0 # Indicateur du tour
        if self.args.eval_type != 'local': 
            self.server_dataset = server_dataset
        self.global_model = self._init_model(model) # Modlele globale
        self.opt_kwargs = dict(lr=self.args.lr, momentum=self.args.beta1) # federation algorithm arguments
        self.curr_lr = self.args.lr # Taux d'apprentissage
        self.clients = self._create_clients(client_datasets) # Liste de nos differents clients
        self.results = defaultdict(dict) 

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [ {str(self.round).zfill(4)}] Initialisation du modele')
        return model
    
    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..optimizer.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        optimizer = ALGORITHM_CLASS(params=model.parameters(), **kwargs)
        return optimizer

    def _create_clients(self, client_datasets):
        CLIENT_CLASS = import_module(f'..clients.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLIENT_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [ {str(self.round).zfill(4)}] Creation des clients ')

        clients = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets), 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [ {str(self.round).zfill(4)}] ...Creation du client... ',
                total=len(client_datasets)
            ):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())            
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [{str(self.round).zfill(4)}] ...Creation de {self.args.K} clients reussit !')
        return clients

    def _sample_clients(self, exclude=[]):
        """ Fonction qui doit selectionner de maniere aleatoire les clients"""

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [ {str(self.round).zfill(4)}] Clients aleatoire ')

        if exclude == []: 
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))
        else: # Sélectionner aléatoirement des clients non participants en quantité égale à `eval_fraction` multipliée
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0: 
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                sampled_client_ids = sorted(random.sample([identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude], num_sampled_clients))
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [ {str(self.round).zfill(4)}] ...{num_sampled_clients} clients on ete selectionne ')
        return sampled_client_ids

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour: {str(self.round).zfill(4)}] [{"EVALUATION" if eval else "MISE A JOUR"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: 
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: 
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            num_samples.append(resulting_sizes[identifier])

            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour: {str(self.round).zfill(4)}] [{"EVALUATION" if eval else "MISE A JOUR "}] [SOMMAIRE] ({len(resulting_sizes)} clients):'

        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()
        
        top10_indices = np.argpartition(losses_array, -int(0.1 * len(losses_array)))[-int(0.1 * len(losses_array)):] if len(losses_array) > 1 else 0
        top10 = np.atleast_1d(losses_array[top10_indices])
        top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

        bot10_indices = np.argpartition(losses_array, max(1, int(0.1 * len(losses_array)) - 1))[:max(1, int(0.1 * len(losses_array)))] if len(losses_array) > 1 else 0
        bot10 = np.atleast_1d(losses_array[bot10_indices])
        bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float), 
            'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
            'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        # self.writer.add_scalars(
        #     f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})',
        #     {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
        #     self.round
        # )

        for name, val in metrics.items():
            val_array = np.array(val).astype(float)
            weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()
            
            top10_indices = np.argpartition(val_array, -int(0.1 * len(val_array)))[-int(0.1 * len(val_array)):] if len(val_array) > 1 else 0
            top10 = np.atleast_1d(val_array[top10_indices])
            top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

            bot10_indices = np.argpartition(val_array, max(1, int(0.1 * len(val_array)) - 1))[:max(1, int(0.1 * len(val_array)))] if len(val_array) > 1 else 0
            bot10 = np.atleast_1d(val_array[bot10_indices])
            bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

            total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
            result_dict[name] = {
                'avg': weighted.astype(float), 'std': std.astype(float), 
                'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
                'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
            }
                
            if save_raw:
                result_dict[name]['raw'] = val

            # self.writer.add_scalars(
            #     f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})',
            #     {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
            #     self.round
            # )
            # self.writer.flush()
        
        logger.info(total_log_string)
        return result_dict

    def _request(self, ids, eval, participated, retain_model, save_raw):
        def __update_clients(client):
            if client.model is None:
                client.download(self.global_model)
            client.args.lr = self.curr_lr
            update_result = client.update()
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client):
            if client.model is None:
                client.download(self.global_model)
            eval_result = client.evaluate() 
            if not retain_model:
                client.model = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [ {str(self.round).zfill(4)}] Requete {"Mise a jour " if not eval else "evaluer"} pour  {"tous les" if ids is None else len(ids)} clients !')
        
        if eval:
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour : {str(self.round).zfill(4)}] ...evaluation des clients... ',
                    total=len(ids)
                ):
                    jobs.append(workhorse.submit(__evaluate_clients, self.clients[idx])) 
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            _eval_sizes, _eval_results = list(map(list, zip(*results)))
            _eval_sizes, _eval_results = dict(ChainMap(*_eval_sizes)), dict(ChainMap(*_eval_results))
            self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                _eval_sizes, 
                _eval_results, 
                eval=True, 
                participated=participated,
                save_raw=save_raw
            )
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour : {str(self.round).zfill(4)}] ...evaluation complete de {"tous les " if ids is None else len(ids)} clients!')
            return None
        else:
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour : {str(self.round).zfill(4)}] ...mises a jour des clients... ',
                    total=len(ids)
                ):
                    jobs.append(workhorse.submit(__update_clients, self.clients[idx])) 
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            update_sizes, _update_results = list(map(list, zip(*results)))
            update_sizes, _update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*_update_results))
            self.results[self.round]['clients_updated'] = self._log_results(
                update_sizes, 
                _update_results, 
                eval=False, 
                participated=True,
                save_raw=False
            )
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour: {str(self.round).zfill(4)}] ...mises a jour complete de  {"tous les" if ids is None else len(ids)} clients!')
            return update_sizes
    
    def _aggregate(self, server_optimizer, ids, updated_sizes):
        assert set(updated_sizes.keys()) == set(ids)

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour: {str(self.round).zfill(4)}] Agregation ')
        
        coefficients = {identifier: float(nuemrator / sum(updated_sizes.values())) for identifier, nuemrator in updated_sizes.items()}
        
        for identifier in ids:
            local_layers_iterator = self.clients[identifier].upload()
            server_optimizer.accumulate(coefficients[identifier], local_layers_iterator)
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour : {str(self.round).zfill(4)}] ...agregation complete du modele!')
        return server_optimizer

    @torch.no_grad()
    def _central_evaluate(self):
        metrics = MetricManager(self.args.eval_metrics)
        self.global_model.eval()
        self.global_model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.global_model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            metrics.track(loss.item(), outputs, targets)
        else:
            self.global_model.to('cpu')
            metrics.aggregate(len(self.server_dataset))

        result = metrics.results
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour : {str(self.round).zfill(4)}] [EVALUATION] [SERVEUR] '

        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '
        
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # log TensorBoard
        # self.writer.add_scalar('Server Loss', loss, self.round)
        # for name, value in result['metrics'].items():
        #     self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        # else:
        #     self.writer.flush()
        # self.results[self.round]['server_evaluated'] = result

    def update(self):
        """Mise a jour du modèle globale.
        """
       
        selected_ids = self._sample_clients() 
        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False) # Requete pour mettre a jour les clients selectionne
        _ = self._request(selected_ids, eval=True, participated=True, retain_model=True, save_raw=False) # Requet pour evaluer les differents cliet selectionne 
       
        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer = self._aggregate(server_optimizer, selected_ids, updated_sizes) # Agregation des mises a jours locales
        server_optimizer.step() # Mises a jour du modele avec les nouvelles valeurs de l'optimisateur
        return selected_ids

    def evaluate(self, excluded_ids):
        """Evaluation du modele global.
        """
       
        if self.args.eval_type != 'global': 
            selected_ids = self._sample_clients(exclude=excluded_ids)
            _ = self._request(selected_ids, eval=True, participated=False, retain_model=False, save_raw=self.round == self.args.R)
        if self.args.eval_type != 'local':
            self._central_evaluate()

        if (not self.args.train_only) and (not self.args.eval_type == 'global'):
            gen_gap = dict()
            curr_res = self.results[self.round]
            for key in curr_res['clients_evaluated_out'].keys():
                for name in curr_res['clients_evaluated_out'][key].keys():
                    if 'avg' in name:
                        gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                        gen_gap[f'gen_gap_{key}'] = {name: gap}
                        # self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
                        # self.writer.flush()
            else:
                self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Sauvegarde du resultat.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour: {str(self.round).zfill(4)}] Sauvegarde du modele dans le checkpoints !')
        
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file: 
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)
        torch.save(self.global_model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt')) 
        
        self.writer.close()
        
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour: {str(self.round).zfill(4)}] ...processus Federated Learning terminé !')
        if self.args.use_tb:
            input('[FINISH] ...Appuyez sur <Enterr> pour quitter !')
