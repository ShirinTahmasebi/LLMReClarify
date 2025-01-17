
# import torch
# import random
# import numpy as np
# import pandas as pd
# import torch.utils.data as data_utils
# from transformers import AutoTokenizer
# import os
# import pickle
# from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
# from config import ROOT_PATH

# os.environ['HUGGINGFACE_TOKEN'] = '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'

# llm_negative_sample_size = 19
# llm_base_model='meta-llama/Llama-2-7b-hf'
# llm_base_tokenizer='meta-llama/Llama-2-7b-hf'
# llm_max_title_len=32
# llm_max_text_len=1536
# llm_train_on_inputst=False
# llm_system_template="Given user history in chronological order, recommend an item from the candidate pool with its index letter."
# llm_input_template ='User history: {}; \n Candidate pool: {}'
# llm_load_in_4bit=True
# llm_retrieved_path=f'{ROOT_PATH}/experiments/lru/ml-1m/'
# llm_cache_dir=None


# def fetch_candidates(gt_pos, scores_all_users, gt_all_users, candidate_size=20):
#     gt_index = gt_pos - 1
#     candidates_all_users = []
#     for u, (p,l) in enumerate(zip(scores_all_users, gt_all_users), start=1):
#         candidates_indices = torch.topk(torch.tensor(p), candidate_size).indices
        
#         if l in candidates_indices:
#             mask = candidates_indices != l
#             candidates_indices = candidates_indices[mask]

#         part1 = candidates_indices[:gt_index]
#         part2 = candidates_indices[gt_index:]
#         candidates_indices = torch.cat((part1, torch.tensor([l]), part2))

#         top_recs = candidates_indices[:candidate_size]
#         candidates_all_users.append(top_recs)
        
#     return candidates_all_users


# #####################
# # AbstractDataset
# def densify_index(df):
#     print('Densifying index')
#     umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
#     smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
#     df['uid'] = df['uid'].map(umap)
#     df['sid'] = df['sid'].map(smap)
#     return df, umap, smap

# # ML100KDataset
# def load_ratings_df():
#     file_path = f'{ROOT_PATH}/data/ml-1m/ratings.dat'
#     df = pd.read_csv(file_path, delimiter="::")
#     df.columns = ['uid', 'sid', 'rating', 'timestamp']
#     return df
    
# # ML1MDataset
# import re
# def load_meta_dict():
#     file_path = f"{ROOT_PATH}/data/ml-1m/movies.dat"
#     df = pd.read_csv(file_path, delimiter='::', header=None, encoding="ISO-8859-1")
#     meta_dict = {}
#     for row in df.itertuples():
#         title = row[2][:-7]  # remove year (optional)
#         year = row[2][-7:]

#         title = re.sub('\(.*?\)', '', title).strip()
#         # the rest articles and parentheses are not considered here
#         if any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
#             title_pre = title.split(', ')[:-1]
#             title_post = title.split(', ')[-1]
#             title_pre = ', '.join(title_pre)
#             title = title_post + ' ' + title_pre

#         meta_dict[row[1]] = title + year
#     return meta_dict
    
# # AbstractDataset
# def split_df(df, user_count):
#     print('Splitting')
#     user_group = df.groupby('uid')
#     user2items = user_group.apply(
#         lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
#     train, val, test = {}, {}, {}
#     for i in range(user_count):
#         user = i + 1
#         items = user2items[user]
#         train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
#     return train, val, test


# # AbstractDataset
# def filter_triplets(df):
#     min_uc = 5
#     min_sc = 5
#     print('Filtering triplets')
#     if min_sc > 1 or min_uc > 1:
#         item_sizes = df.groupby('sid').size()
#         good_items = item_sizes.index[item_sizes >= min_sc]
#         user_sizes = df.groupby('uid').size()
#         good_users = user_sizes.index[user_sizes >= min_uc]
#         while len(good_items) < len(item_sizes) or len(good_users) < len(user_sizes):
#             if min_sc > 1:
#                 item_sizes = df.groupby('sid').size()
#                 good_items = item_sizes.index[item_sizes >= min_sc]
#                 df = df[df['sid'].isin(good_items)]

#             if min_uc > 1:
#                 user_sizes = df.groupby('uid').size()
#                 good_users = user_sizes.index[user_sizes >= min_uc]
#                 df = df[df['uid'].isin(good_users)]

#             item_sizes = df.groupby('sid').size()
#             good_items = item_sizes.index[item_sizes >= min_sc]
#             user_sizes = df.groupby('uid').size()
#             good_users = user_sizes.index[user_sizes >= min_uc]
#     return df

# # dataloader.llm
# def generate_and_tokenize_eval(data_point, tokenizer, prompter):
#     in_prompt = prompter.generate_prompt(data_point["system"],
#                                          data_point["input"])
#     tokenized_full_prompt = tokenizer(in_prompt,
#                                       truncation=True,
#                                       max_length=llm_max_text_len,
#                                       padding=False,
#                                       return_tensors=None)
#     tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord('A')
    
#     return tokenized_full_prompt  


# # dataloader.llm
# def seq_to_token_ids(seq, candidates, label, text_dict, tokenizer, prompter, eval=False):
#     def truncate_title(title):
#         title_ = tokenizer.tokenize(title)[:llm_max_title_len]
#         title = tokenizer.convert_tokens_to_string(title_)
#         return title

#     seq_t = ' \n '.join(['(' + str(idx + 1) + ') ' + truncate_title(text_dict[item]) 
#                        for idx, item in enumerate(seq)])
#     can_t = ' \n '.join(['(' + chr(ord('A') + idx) + ') ' + truncate_title(text_dict[item])
#                        for idx, item in enumerate(candidates)])
#     output = chr(ord('A') + candidates.index(label))  # ranking only
    
#     data_point = {}
#     data_point['system'] = llm_system_template if llm_system_template is not None else DEFAULT_SYSTEM_PROMPT
#     data_point['input'] = llm_input_template.format(seq_t, can_t)
#     data_point['output'] = output
    
#     return generate_and_tokenize_eval(data_point, tokenizer, prompter)
    
# # trainer.llm
# def llama_collate_fn_w_truncation(llm_max_length, eval=False):
#     def llama_collate_fn(batch):
#         all_input_ids = []
#         all_attention_mask = []
#         all_labels = []
#         example_max_length = max([len(batch[idx]['input_ids']) for idx in range(len(batch))])
#         max_length = min(llm_max_length, example_max_length)
        
#         for i in range(len(batch)):
#             input_ids = batch[i]['input_ids']
#             attention_mask = batch[i]['attention_mask']
#             labels = batch[i]['labels']
#             if len(input_ids) > max_length:
#                 input_ids = input_ids[-max_length:]
#                 attention_mask = attention_mask[-max_length:]
#                 if not eval: labels = labels[-max_length:]
#             elif len(input_ids) < max_length:
#                 padding_length = max_length - len(input_ids)
#                 input_ids = [0] * padding_length + input_ids
#                 attention_mask = [0] * padding_length + attention_mask
#                 if not eval: labels = [-100] * padding_length + labels

#             if eval: assert input_ids[-1] == 13
#             else:
#                 assert input_ids[-3] == 13 and input_ids[-1] == 2
#                 assert labels[-3] == -100 and labels[-2] != -100
            
#             all_input_ids.append(torch.tensor(input_ids).long())
#             all_attention_mask.append(torch.tensor(attention_mask).long())
#             all_labels.append(torch.tensor(labels).long())
        
#         return {
#             'input_ids': torch.vstack(all_input_ids),
#             'attention_mask': torch.vstack(all_attention_mask),
#             'labels': torch.vstack(all_labels)
#         }
#     return llama_collate_fn

  
# def convert_u2seq_to_all_seqs(u2seq):
#     all_seqs = []
#     for u in sorted(u2seq.keys()):
#         seq = u2seq[u]
#         for i in range(2, len(seq)+1):
#             all_seqs += [seq[:i]]
#     return all_seqs


# retrieved_file = pickle.load(open(os.path.join(llm_retrieved_path, 'retrieved.pkl'), 'rb'))
        
# print('******************** Constructing Validation Subset ********************')
# val_probs = retrieved_file['val_probs']
# val_labels = retrieved_file['val_labels']
# val_metrics = retrieved_file['val_metrics']
# val_candidates_injected = fetch_candidates(1, val_probs, val_labels)

# # If the ground trith label is within the top k retreived items, then select the user!
# # What the fuck?!
# val_users = [u for u, (p, l) in enumerate(zip(val_probs, val_labels), start=1) \
#                     if l in torch.topk(torch.tensor(p), llm_negative_sample_size+1).indices]
# val_candidates = [torch.topk(torch.tensor(val_probs[u-1]), 
#                         llm_negative_sample_size+1).indices.tolist() for u in val_users]
# print(f"The perfromance of retriever for validation sets is: {round(len(val_users) / len(val_labels), 2)}")


# print('******************** Constructing Test Subset ********************')
# test_probs = retrieved_file['test_probs']
# test_labels = retrieved_file['test_labels']
# test_metrics = retrieved_file['test_metrics']
# test_candidates_injected = fetch_candidates(1, test_probs, test_labels)

# test_users = [u for u, (p, l) in enumerate(zip(test_probs, test_labels), start=1) \
#                     if l in torch.topk(torch.tensor(p), llm_negative_sample_size+1).indices]
# test_candidates = [torch.topk(torch.tensor(test_probs[u-1]), 
#                         llm_negative_sample_size+1).indices.tolist() for u in test_users]
# non_test_users = [u for u, (p, l) in enumerate(zip(test_probs, test_labels), start=1) \
#                         if l not in torch.topk(torch.tensor(p), llm_negative_sample_size+1).indices]
# print(f"The perfromance of retriever for test sets is: {round(len(test_users) / len(test_labels), 2)}")



# df = load_ratings_df()
# df = filter_triplets(df)
# df, umap, smap = densify_index(df)
# train, val, test = split_df(df, len(umap))
# meta_raw = load_meta_dict()
# text_dict = meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
# user_count = len(umap)
# item_count = len(smap)
# dataset = {
#     'train': train,
#     'val': val,
#     'test': test,
#     'meta': meta,
#     'umap': umap,
#     'smap': smap
# }

# args = {
#     'num_items': item_count,
#     'llm_max_history': llm_max_history,
#     'llm_base_tokenizer': llm_base_tokenizer,
#     'llm_cache_dir': llm_cache_dir,
#     'llm_retrieved_path': llm_retrieved_path,
#     'llm_negative_sample_size': llm_negative_sample_size
# }

# from dataloader.llm import LLMDataloader

# dataloader = LLMDataloader(args, dataset)
# train, val, test = dataloader.get_pytorch_dataloaders()
# tokenizer = dataloader.tokenizer
# test_retrieval = dataloader.test_retrieval
    
    
# for user_id in range(1,10): #len(val_users)):
#     user_index = user_id - 1
#     gt_id_conv = val_labels[user_index]
#     candiates_injected = val_candidates_injected[user_index]
#     candiates = val_candidates[user_index]
    
#     user_id_org = next((k for k, v in umap.items() if v == user_id), None)
#     gt_id_org = next((k for k, v in smap.items() if v == gt_id_conv), None)
#     candiates_injected_org = [str(next((k for k, v in smap.items() if v == d), None)) for d in candiates_injected]
#     candiates_org = [str(next((k for k, v in smap.items() if v == d), None)) for d in candiates]

#     print(f"User {user_id_org} has wateched movie {gt_id_org}")
#     print(f"Candidate Items: {', '.join(candiates_org)}")
#     print(f"Candidate Items with GT: {', '.join(candiates_injected_org)}")
#     print("-----------------------------------------------------------")
    
    
# # # From LLMDataLoader
# # tokenizer = AutoTokenizer.from_pretrained(
# #     llm_base_tokenizer, 
# #     cache_dir=llm_cache_dir, 
# #     use_auth_token=os.environ['HUGGINGFACE_TOKEN']
# # )
# # tokenizer.pad_token = tokenizer.unk_token
# # tokenizer.padding_side = 'left'
# # tokenizer.truncation_side = 'left'
# # tokenizer.clean_up_tokenization_spaces = True
# # pass


hr10_fixed = [1.0, 0.9640728235244751, 1.0, 0.4210264980792999, 0.99701988697052, 0.026324503123760223, 0.7743377685546875, 0.11556291580200195, 0.812251627445221, 0.06589403748512268, 0.029139073565602303, 0.06423841416835785, 0.2705298066139221, 0.045198675245046616, 0.9503311514854431, 0.2947019934654236, 0.22052979469299316, 0.8261589407920837, 0.7387086272239685, 0.3862582743167877]

hr10_random = [0.32036423683166504, 0.32036423683166504, 0.32036423683166504]

## --------------------------------

ndcg10_fixed = [0.9982523322105408, 0.3837221562862396, 0.6105840802192688, 0.1479453146457672, 0.43275606632232666, 0.010505584068596363, 0.2676367163658142, 0.04150690510869026, 0.26085183024406433, 0.02640349231660366, 0.011767576448619366, 0.023534081876277924, 0.09714774042367935, 0.01646268554031849, 0.3373015820980072, 0.10262426733970642, 0.06814418733119965, 0.30655381083488464, 0.2654802918434143, 0.13343043625354767]

ndcg10_random = [0.1895863115787506, 0.1895863115787506, 0.1895863115787506]

## --------------------------------

hr5_fixed = [1.0, 0.570364236831665, 1.0, 0.10894040018320084, 0.8370860815048218, 0.013576159253716469, 0.18609271943569183, 0.03460264950990677, 0.06341059505939484, 0.028807947412133217, 0.014569536782801151, 0.021192053332924843, 0.08327814936637878, 0.015231788158416748, 0.29735100269317627, 0.07301324605941772, 0.007284768391400576, 0.32036423683166504, 0.233907288312912, 0.0910596027970314]

hr5_random = [0.2256622463464737, 0.2256622463464737, 0.2256622463464737]

## --------------------------------

ndcg5_fixed = [0.9982523322105408, 0.2540796995162964, 0.6105840802192688, 0.050046294927597046, 0.37822654843330383, 0.006528351455926895, 0.0812285840511322, 0.016278499737381935, 0.02629593014717102, 0.014800774864852428, 0.007129892241209745, 0.009941397234797478, 0.03843177855014801, 0.007148026488721371, 0.12168540060520172, 0.0334232822060585, 0.0029323764611035585, 0.1438918113708496, 0.10468347072601318, 0.041796714067459106]

ndcg5_random = [0.158999517560005, 0.1589995175600052, 0.1589995175600052]



def print_performance(fixed_arr, random_arr, caption):
    import numpy as np
    half_len = int(len(fixed_arr) / 2)
    weights = list(range(1, half_len + 1)) + list(range(half_len, 0, -1))
    
    fixed_avg_simple = np.average(fixed_arr)
    fixed_avg_weighted = np.average(fixed_arr, weights=weights)
    random_avg = np.average(random_arr)
    
    print("-----------------------------------")
    print(f"{weights}")
    print(f"{caption} AVG Simple: {fixed_avg_simple}")
    print(f"{caption} AVG Weighted: {fixed_avg_weighted}")
    print(f"{caption} Random: {random_avg}")




print_performance(hr10_fixed, hr10_random, "HR@10")
print_performance(ndcg10_fixed, ndcg10_random, "NDCG@10")

print_performance(hr5_fixed, hr5_random, "HR@5")
print_performance(ndcg5_fixed, ndcg5_random, "NDCG@5")

