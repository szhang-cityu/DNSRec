from recbole.quick_start.quick_start import run_recbole
import argparse

parser = argparse.ArgumentParser(description='hyper_flops')
parser.add_argument('--hyperflops', type=float, default=0.01, help='')
args = parser.parse_args()
hyper_flops = args.hyperflops

parameter_dict = {
   # ‘neg_sampling': {'popularity': 100},
   'hyper_flops': hyper_flops,
   'decide_f': 1, # 'Decide funtion to use. 1: Hand select--gumbel-softmax, 2: auto select by argmax'
   'softmax_type':1, # '0 softmax; 1 softmax+temperature; 2 gumbel-softmax'
   'options': [0,1,2], # three options
   'mask_size': [0, 32, 48], #0: auto-dim-16, no mask ;1: auto-dim-8 8 dim set as 0; 2: auto-dim-4, 12 dim set as 0
   'mask_ffn': [0, 128, 192], # auto-dim for inner size
   # 'weight': [0.5,0.5], #Initial weights for options
   'darts_frequency': 10,
   'learning_rate': 0.001,    #0.001
   'weight_decay': 1e-6,
   'train_batch_size': 1024,   #512 for gowalla
   'eval_batch_size': 1024,
   'train_neg_sample_args': None,
   'neg_sampling': None,
   'mask_ratio': 0.2,
   #'emb_size': 128,
   'hidden_size': 64,
   'inner_size': 256,
   'n_layers': 4,
   'n_heads': 4,
   'hidden_dropout_prob': 0.2,
   'attn_dropout_prob': 0.2,
   'hidden_act': 'gelu',
   'layer_norm_eps': 1e-12,
   'top_k': 30, # Select top-k attention probs for sparse attention block
   'initializer_range': 0.02,
   # 'loss_type': 'CE',
   'eval_args': {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'pop100', 'group_by': 'user'},
   'topk': 10,
   'metrics': ['Recall', 'MRR', 'NDCG'],
   'valid_metric': 'NDCG@10'
}
# print(run_recbole(model='BERT4Rec', dataset='gowalla-merged', config_file_list=['gowalla.yaml'], config_dict=parameter_dict))
run_recbole(model='SASRec', dataset='gowalla', config_dict=parameter_dict)


# FDSA, SASRecF
# from recbole.quick_start import run_recbole
# parameter_dict = {
#       # 'neg_sampling': {'popularity': 100},
#    'learning_rate': 0.001,    #0.001
#    #'weight_decay': 0.01,
#    'train_batch_size': 1024,   #2048
#    'eval_batch_size': 1024,   #2048
#    'neg_sampling': None,
#    'mask_ratio': 0.2,
#    'hidden_size': 64,   #128
#    'inner_size': 256,
#    'n_layers': 2,
#    'n_heads': 8,
#    'hidden_dropout_prob': 0.2,
#    'attn_dropout_prob': 0.2,
#    'hidden_act': 'gelu',
#    'layer_norm_eps': 1e-12,
#    'initializer_range': 0.02,
#    # 'loss_type': 'CE',
#    # 'eval_args': {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'pop100', 'group_by': 'user'},
#    'topk': 10,
#    'metrics': ['Recall', 'MRR', 'NDCG'],
#    'valid_metric': 'NDCG@10',
#    'train_neg_sample_args': None,
#    # 'load_col':
#    #    {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
#    #    'item': ['item_id', 'genre']},
#    # 'selected_features': ['genre'],
#
#    # # gowalla
#    # 'load_col':
#    #    {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
#    # 'selected_features': ['item_id'],
# }
# run_recbole(model='SASRec', dataset='gowalla', config_dict=parameter_dict)

