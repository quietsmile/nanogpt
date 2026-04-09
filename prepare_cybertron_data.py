"""
Prepare training/validation data for nanogpt by extracting it from cybertron's
blended megatron dataset, preserving the EXACT same data order as cybertron.

This script replicates the data pipeline from the scaling_moe_00196 experiment
(PAI job dlc1q9arre48b0kx) using the same:
  - Data paths and blending weights (from data_pretrain_v3_pai.yaml)
  - Random seed: 1234 (default gpt_dataset_seed)
  - Sequence length: 8192
  - Cache path: /prodcpfs/user/data/save/data/lossalign/data_cache

Output:
  data/cybertron_baseline/train.bin   -- uint16 token array, sequential samples
  data/cybertron_baseline/val.bin     -- validation tokens (megatron val sets)
  data/cybertron_baseline/meta.pkl    -- vocab_size=152064

Usage:
  python prepare_cybertron_data.py [--n_train_samples N] [--n_val_samples N]

The output format matches nanogpt's expectations:
  - Each sample is exactly seq_len tokens
  - Samples are stored consecutively (no gap/stride)
  - Dtype: uint16 (sufficient for vocab_size=152064)

Notes:
  - Run this once before training with config/cybertron_baseline.py
  - The blended dataset indices are cached at data_cache_path; building
    them from scratch requires several hours and ~50GB of memory.
  - If the cache already exists (from the cybertron run), this is fast.
"""

import os
import sys
import pickle
import argparse
import numpy as np

# Paths
CYBERTRON_PATH = '/newcpfs/user/yuchen/llm/cybertron_dots3.0_swa'
MEGATRON_PATH  = '/newcpfs/user/yuchen/llm/megatron_dots3.0_swa'
sys.path.insert(0, CYBERTRON_PATH)
sys.path.insert(0, MEGATRON_PATH)

# Experiment parameters (from scaling_moe_00196.yaml + data_pretrain_v3_pai.yaml)
SEQ_LENGTH     = 8192
RANDOM_SEED    = 1234   # cybertron default gpt_dataset_seed
DATA_CACHE_PATH = '/prodcpfs/user/data/save/data/lossalign/data_cache'
VOCAB_SIZE     = 152064  # Qwen tokenizer padded vocab
EOD_TOKEN_ID   = 151643  # Qwen <|endoftext|> token id

# Training data: path → (weight, loss_group) from data_pretrain_v3_pai.yaml
# Abbreviated to just paths+weights here; loss_group is not needed for data extraction
TRAIN_DATA_CONFIG = [
    (0.0029135789249514323, '/prodcpfs/data/text_data/pretrain_v3/business_zh_quiz_v0_pos_qwen_with_st'),
    (0.024630292151806604,  '/prodcpfs/data/text_data/pretrain_v3/zh_20250117_domain_repair_qwen_with_st'),
    (0.033960756072404334,  '/prodcpfs/data/text_data/pretrain_v3/vertical_domain_zh_all_qwen_with_st'),
    (0.019683320551727897,  '/prodcpfs/data/text_data/pretrain_v3/zh_v3_20250425_merge_filter_semdedup_balance_repart_rule_filter_part0_qwen_with_st'),
    (0.019676501748809118,  '/prodcpfs/data/text_data/pretrain_v3/zh_v3_20250425_merge_filter_semdedup_balance_repart_rule_filter_part1_qwen_with_st'),
    (0.019678774720152854,  '/prodcpfs/data/text_data/pretrain_v3/zh_v3_20250425_merge_filter_semdedup_balance_repart_rule_filter_part2_qwen_with_st'),
    (0.019676785863282562,  '/prodcpfs/data/text_data/pretrain_v3/zh_v3_20250425_merge_filter_semdedup_balance_repart_rule_filter_part3_qwen_with_st'),
    (0.001746540280396514,  '/prodcpfs/data/text_data/data_version_0121/zh_web_question_merge_dec_new_qwen_with_st'),
    (0.0022332523181273424, '/prodcpfs/data/text_data/data_version_0121/business_zh_qa_dec_qwen_with_st'),
    (0.0007530968569064392, '/prodcpfs/data/text_data/data_version_0121/zh_qa_0122_cat_need_label_v2_qwen_with_st'),
    (0.004974781484748026,  '/prodcpfs/data/text_data/dense_pretrain_v1_dec/crawl_page_cn_question_v1_dec_qwen_with_st'),
    (0.000553056989551299,  '/prodcpfs/data/text_data/red_ct_v241202/zh_select_source_wo_minhash_red_ct_v241202_qwen_with_st'),
    (0.004042683287242916,  '/prodcpfs/data/text_data/dense_pretrain_v1_dec/xiaohongshu_note_241125_clean_v2_basic_high_dec_qwen_with_st'),
    (0.00011532880929362725,'/prodcpfs/data/text_data/red_ct_v241202/xiaohongshu_note_goods_241125_clean_v1_red_ct_v241202_qwen_with_st'),
    (0.003943691060172098,  '/prodcpfs/data/text_data/dense_pretrain_v1_dec/zh_direct_logic_from_chuilei_dec_qwen_with_st'),
    (0.0010125067046484425, '/prodcpfs/data/text_data/data_version_0121/open_source_part_data_rednote_recall_0117_qwen_with_st'),
    (0.00021142416874837298,'/prodcpfs/data/text_data/data_version_0121/open_source_part_data_zhneweval_recall_0121_qwen_with_st'),
    (0.00016131735354122176,'/prodcpfs/data/text_data/data_version_0121/open_source_part_data_zhbookeval_recall_0121_qwen_with_st'),
    (0.0009100090472040127, '/prodcpfs/data/text_data/data_version_0212_anneal/zhv2eval_recall_0206_hq_cat_filter_qwen_with_st'),
    (0.0015757790373145905, '/prodcpfs/data/text_data/data_version_0212_anneal/ownthink_cat_edu_retrival_web_cn_v2_knowledge_explode_top30_hq_cat_filter_qwen_with_st'),
    (0.0007280504493857055, '/prodcpfs/data/text_data/data_version_0212_anneal/baai_industry_corpus2_hf_zh_v1_dec_corners_qwen_with_st'),
    (0.0005100823698606325, '/prodcpfs/data/text_data/data_version_0121/query_retrieve_res_zh_eval_res115_repart_qwen_with_st'),
    (0.002984422430009656,  '/prodcpfs/data/text_data/data_version_0121/open_source_part_data_zhgoogle_recall_0117_qwen_with_st'),
    (0.0027838327228141492, '/prodcpfs/data/text_data/pretrain_v3/zh_zhiyi_tiku_old_20250121_cpoy2_filter_qwen_with_st'),
    (0.002017875047040754,  '/prodcpfs/data/text_data/data_version_0121/new_tiku_clean_v1_qwen_with_st'),
    (0.00018480418704975123,'/prodcpfs/data/text_data/data_version_0121/open_source_part_data_zh_zhiye_tiku_old_202501_qwen_with_st'),
    (0.0019185968165852535, '/prodcpfs/data/text_data/pretrain_v3/zh_k12_tiku_old_202501_filter_qwen_with_st'),
    (0.00047740637204814825,'/prodcpfs/data/text_data/pretrain_v3/tt_qa_qwen_with_st'),
    (0.0009905734450811098, '/prodcpfs/data/text_data/data_version_0121/open_source_part_data_zh_common_crawl_hq_20250121_qwen_with_st'),
    (0.00019156976711644474,'/prodcpfs/data/text_data/data_version_0121/baike_sogou_clean_v1_qwen_with_st'),
    (0.00021275190668359914,'/prodcpfs/data/text_data/data_version_0121/baike_bytedance_clean_v1_qwen_with_st'),
    (2.8000331367588198e-05,'/prodcpfs/data/text_data/data_version_0121/baidu_edu_inc_clean_v1_qwen_with_st'),
    (0.00048533991038139093,'/prodcpfs/data/text_data/data_version_0121/gwyoo_article_clean_v1_qwen_with_st'),
    (7.286186227763726e-05, '/prodcpfs/data/text_data/data_version_0121/intern_hrbfnkj_qwen_with_st'),
    (5.814702146941331e-05, '/prodcpfs/data/text_data/data_version_0121/intern_1mpi_qwen_with_st'),
    (8.448822209104586e-06, '/prodcpfs/data/text_data/data_version_0121/intern_babytree_ask_qwen_with_st'),
    (1.0000118345567214e-06,'/prodcpfs/data/text_data/data_version_0121/intern_babytree_common_qwen_with_st'),
    (3.7111550304660548e-06,'/prodcpfs/data/text_data/data_version_0121/intern_babytree_topic_qwen_with_st'),
    (1.7100480151985092e-05,'/prodcpfs/data/text_data/data_version_0121/intern_hupu_qwen_with_st'),
    (0.0003954676801293047, '/prodcpfs/data/text_data/data_version_0121/intern_meipian_qwen_with_st'),
    (0.00043914680815741654,'/cpfs/user/guofu/data/mathproblem/bins/jiaoyubu_problems_0424_qwen_with_st'),
    (0.0075985415354176975, '/prodcpfs/data/text_data/pretrain_v3/knowledge_recall_dense_pretrain_zh_repair_qwen_with_st'),
    (0.006683921100307266,  '/prodcpfs/data/text_data/pretrain_v3/zh_v3_20250425_merge_filter_semdedup_balance_renwen_rebalance_inc_qwen_with_st'),
    (0.0003105620086535445, '/prodcpfs/data/text_data/dense_pretrain_v1/business_en_normal_v0_qwen_with_st'),
    (0.009423922859963088,  '/prodcpfs/data/text_data/data_version_0121/en20250117_domain_v2_split_otherall_qwen_with_st'),
    (0.007115569653060537,  '/prodcpfs/data/text_data/pretrain_v3/en20250117_domain_v2_split_pile_new_clean_repair_qwen_with_st'),
    (0.003006565136500115,  '/prodcpfs/data/text_data/dense_pretrain_v1_dec/business_en_quiz_v0_dec_qwen_with_st'),
    (0.026889516999525905,  '/prodcpfs/data/text_data/pretrain_v3/domain_enhance_en_v1_qwen_with_st'),
    (0.0008089231286889115, '/prodcpfs/data/text_data/data_version_0121/business_en_qa_v0_cat_need_label_qwen_with_st'),
    (0.0006994106104499182, '/prodcpfs/data/text_data/data_version_0121/open_source_part_data_enbookeval_recall_0121_qwen_with_st'),
    (0.002058751030784177,  '/prodcpfs/data/text_data/data_version_0212_anneal/env2eval_recall_0206_hq_cat_filter_qwen_with_st'),
    (0.002257519383091678,  '/prodcpfs/data/text_data/data_version_0212_anneal/ownthink_cat_edu_retrival_web_en_v2_knowledge_explode_top30_hq_cat_filter_qwen_with_st'),
    (0.009508797864409255,  '/prodcpfs/data/text_data/data_version_0212_anneal/baai_industry_corpus2_hf_en_v1_dec_corners_qwen_with_st'),
    (0.0064618971394498695, '/prodcpfs/data/text_data/data_version_0212_anneal/en_web_page_v1_reformat_question_filter_qwen_with_st'),
    (3.9129351961943894e-06,'/prodcpfs/data/text_data/data_version_0121/open_source_part_data_enneweval_recall_0121_qwen_with_st'),
    (0.0002254225566333501, '/prodcpfs/data/text_data/data_version_0121/open_source_part_data_engoogle_recall_0117_qwen_with_st'),
    (0.0022925052415733497, '/prodcpfs/data/text_data/data_version_0121/query_retrieve_res_en_eval_res115_repart_qwen_with_st'),
    (0.00026400779104486906,'/prodcpfs/data/text_data/data_version_0121/open_source_part_data_en_common_crawl_hq_20250121_qwen_with_st'),
    (0.005031559434459717,  '/prodcpfs/data/text_data/data_version_0121/open_source_part_data_en_news_qwen_with_st'),
    (0.0002508514131245887, '/prodcpfs/data/text_data/data_version_0121/ycombinator_qwen_with_st'),
    (0.0003089006556590676, '/prodcpfs/data/text_data/data_version_0121/physicsforums_qwen_with_st'),
    (4.211160947744416e-05, '/prodcpfs/data/text_data/data_version_0121/livelaptopspec_qwen_with_st'),
    (0.00010635614755185405,'/prodcpfs/data/text_data/data_version_0121/intern_boards_qwen_with_st'),
    (0.00037961738143915463,'/prodcpfs/data/text_data/data_version_0121/open_source_part_data_en_6m_tiku_20250121_qwen_with_st'),
    (0.0002580617206765948, '/prodcpfs/data/text_data/data_version_0121/intern_quizlet_qwen_with_st'),
    (0.06266688918163331,   '/prodcpfs/data/text_data/pretrain_v3/en_v3_20250423_merge_filter_semdedup_balance_repart_0_qwen_with_st'),
    (0.06267386326416752,   '/prodcpfs/data/text_data/pretrain_v3/en_v3_20250423_merge_filter_semdedup_balance_repart_1_qwen_with_st'),
    (0.06267386326416752,   '/prodcpfs/data/text_data/pretrain_v3/en_v3_20250423_merge_filter_semdedup_balance_repart_2_qwen_with_st'),
    (0.06267526350296075,   '/prodcpfs/data/text_data/pretrain_v3/en_v3_20250423_merge_filter_semdedup_balance_repart_3_qwen_with_st'),
    (0.0006904935049211759, '/prodcpfs/data/text_data/pretrain_v3/knowledge_recall_dense_pretrain_en_qwen_with_st'),
    (0.0024538391508660542, '/prodcpfs/data/text_data/pretrain_v3/RedStone-QA-oq-hqv8-catv4-filter_qwen_with_st'),
    (0.002461731355376802,  '/prodcpfs/data/text_data/pretrain_v3/en_10m_duotan_entiku_20250411_qwen_with_st'),
    (0.0010491158601186644, '/prodcpfs/data/text_data/pretrain_v3/en_5m_tiku_20250329_repart_qwen_with_st'),
    (0.0011058466426054933, '/prodcpfs/data/text_data/data_version_0121/duotan_quizlet_word_clean_v1_qwen_with_st'),
    (0.0047745165036176835, '/prodcpfs/data/text_data/pretrain_v3/zh_tiku_translate_2en0418_final_qwen_with_st'),
    (0.0025408744030997166, '/prodcpfs/data/text_data/pretrain_v3/chegg_tiku0417_re_qwen_with_st'),
    (0.0005302712754512198, '/prodcpfs/data/text_data/pretrain_v3/chegg_tiku0417_woimg_qwen_with_st'),
    (0.0009860612250149508, '/prodcpfs/data/text_data/pretrain_v3/enquzilet_tiku_20250123_qwen_with_st'),
    (0.00977483445724159,   '/prodcpfs/data/text_data/pretrain_v3/en_v3_20250425_merge_filter_semdedup_balance_renwen_rebalance_inc_qwen_with_st'),
    (0.058814033251997556,  '/prodcpfs/data/dataark/prod/sample/scihub_2_1747661333720_qwen_with_st'),
    (0.0026071148536834757, '/prodcpfs/data/text_data/red_ct_v241202/open-web-math_red_ct_v241202_qwen_with_st'),
    (0.001170097847425467,  '/prodcpfs/data/text_data/red_ct_v241202/open-web-math_filtered_v1_1128_red_ct_v241202_qwen_with_st'),
    (0.0036741693149839093, '/prodcpfs/data/text_data/data_version_0121/math_problem_dedup_healthy_qwen_with_st'),
    (0.002232388335680392,  '/cpfs/user/guofu/data/mathproblem/bins/math_problem_v3_qwen_with_st'),
    (0.010383459965525005,  '/cpfs/user/guofu/data/mathpage/bins/math_article_v3_qwen_with_st'),
    (0.003044071858148065,  '/cpfs/user/guofu/data/mathpage/bins/math_article_zh_v0_qwen_with_st'),
    (0.018488534801024387,  '/cpfs/user/guofu/data/mathpage/bins/megamath_top100b_qwen_with_st'),
    (0.00016693375334137048,'/prodcpfs/data/text_data/pretrain_v3/cn_normal_hq_quiz_v2_processed_qwen_with_st'),
    (0.0004964099858337208, '/prodcpfs/data/text_data/pretrain_v3/business_cn_web_page_merge_logic_basic_parquet_qwen_with_st'),
    (0.0003978174857156609, '/prodcpfs/data/text_data/pretrain_v3/crawl_page_en_v4_parquet_qwen_with_st'),
    (0.001857683873493882,  '/prodcpfs/data/text_data/pretrain_v3/en_normal_hq_quiz_v4_parquet_qwen_with_st'),
    (0.0006367892026954199, '/prodcpfs/data/text_data/pretrain_v3/minhash_zh_inc_logic_processed_qwen_with_st'),
    (0.0007898752366015533, '/prodcpfs/data/text_data/pretrain_v3/minhash_en_inc_logic_processed_qwen_with_st'),
    (0.010324777632164932,  '/prodcpfs/data/text_data/data_version_0212_anneal/agi_pt_moe_no_web_code_en_v2_retrival_logic_unq_no_translated_qwen_with_st'),
    (0.0031319231200387246, '/prodcpfs/data/text_data/data_version_0212_anneal/agi_pt_moe_no_web_code_zh_v2_retrival_logic_unq_no_translated_qwen_with_st'),
    (0.0031241284722380593, '/prodcpfs/data/text_data/pretrain_v3/ocr_books_manticore_search_results_merged_qwen_with_st'),
    (0.011738882367269577,  '/prodcpfs/data/text_data/pretrain_v3/ocr_books_manticore_search_results_merged_en_qwen_with_st'),
    (0.002271060543343411,  '/prodcpfs/data/text_data/pretrain_v3/en_code_related_web_v1_qwen_with_st'),
    (0.007788659757578177,  '/prodcpfs/data/text_data/pretrain_v3/zh_code_related_web_v1_qwen_with_st'),
    (0.0038093679983147942, '/prodcpfs/data/text_data/pretrain_v3/zh_code_related_web_special_domain_qwen_with_st'),
    (0.010159209394983647,  '/prodcpfs/data/text_data/pretrain_v3/stackexchange_stackoverflow_qwen_with_st'),
    (0.007848876053536248,  '/prodcpfs/data/text_data/data_version_0121/python_minhash_filter_quality_concat_exact_dedup_github_full_filter_toplang_clean_copyright_qwen_with_st_0'),
    (0.010551698039857806,  '/prodcpfs/data/text_data/data_version_0121/nonpython_minhash_filter_quality_concat_exact_dedup_github_full_filter_toplang_clean_copyright_qwen_with_st_0'),
    (0.15634060019980464,   '/cpfs/2926428ee2463e44/user/dubi/binidx/github_0705_pre9th_score1_qwen_with_st'),
    (0.001231808411067323,  '/prodcpfs/data/text_data/data_version_0121/code_anneal_0211/opc_anneal_algorithmic_corpus_qwen_with_st_0'),
    (0.0013299054886556797, '/prodcpfs/data/text_data/data_version_0121/minhash_dedup_stage2_zh_translated_from_business_en_quiz_v0_qwen_with_st'),
    (0.0031701180164974887, '/prodcpfs/data/text_data/data_version_0121/minhash_dedup_stage2_en_translated_from_business_zh_quiz_v0_qwen_with_st'),
    (0.0023623104010100254, '/prodcpfs/data/text_data/dense_pretrain_v1_dec/paper_v0_dec_qwen_with_st'),
    (0.004720899757983665,  '/prodcpfs/data/text_data/data_version_0121/en20250117_domain_v2_split_pes2o_v2_qwen_with_st'),
    (0.0010648619353531685, '/prodcpfs/data/text_data/data_version_0121/mdpi_qwen_with_st'),
    (0.018376647921352875,  '/prodcpfs/data/text_data/pretrain_v3/zh_non_duxiu_books_qwen_with_st'),
    (0.03037788822680894,   '/prodcpfs/data/text_data/pretrain_v3/zh_duxiu_books_qwen_with_st'),
    (0.003743745027260767,  '/prodcpfs/data/dataark/prod/sample/zh_books_0514_1_1747661523248_qwen_with_st'),
    (0.04542739205113004,   '/prodcpfs/data/text_data/pretrain_v3/en_books_all_qwen_with_st'),
    (0.0022525869913863667, '/prodcpfs/data/dataark/prod/sample/en_books_0514_1_1750130552716_qwen_with_st'),
    (0.008085936470119104,  '/prodcpfs/data/text_data/pretrain_v3/en_books_all_pdfdrive_search_qwen_with_st'),
]

# Validation data paths (megatron format, subset for loss evaluation)
# Using Pile_test_5k as primary val set (small, fast)
VAL_DATA_PATH = '/cpfs/user/wangzerui/cybertron_workspace/datasets/prod_validation_datasets/OOD_Validation/megatron_bins_raw/Pile_test_5k_gpt'


def init_distributed():
    """Initialize single-process distributed group for megatron dataset build."""
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    return dist


class _MinimalTokenizer:
    """Minimal tokenizer stub that satisfies megatron's GPTDatasetConfig requirement."""
    eod = EOD_TOKEN_ID
    pad = EOD_TOKEN_ID

    def tokenize(self, text):
        raise NotImplementedError

    def detokenize(self, tokens):
        raise NotImplementedError


def build_train_dataset(n_train_samples):
    """Build the blended training dataset using cybertron's pipeline."""
    import torch
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.utils import Split

    weights = [w for w, _ in TRAIN_DATA_CONFIG]
    paths   = [p for _, p in TRAIN_DATA_CONFIG]

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    config = GPTDatasetConfig(
        random_seed=RANDOM_SEED,
        sequence_length=SEQ_LENGTH,
        blend=None,
        blend_per_split=[(paths, weights), None, None],
        split=None,
        path_to_cache=DATA_CACHE_PATH,
        tokenizer=_MinimalTokenizer(),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=True,
        create_attention_mask=False,
    )

    builder = BlendedMegatronDatasetBuilder(
        cls=GPTDataset,
        sizes=[n_train_samples, None, None],
        is_built_on_rank=lambda: True,
        config=config,
    )

    datasets = builder.build()
    return datasets[0]  # train split


def write_bin(samples_iter, out_path, n_samples, seq_len, desc):
    """Write samples to a binary file of uint16."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Pre-allocate the output array
    print(f"Writing {n_samples:,} samples ({n_samples * seq_len:,} tokens) to {out_path}")
    out = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(n_samples * seq_len,))
    for i, tokens in enumerate(samples_iter):
        if i >= n_samples:
            break
        if i % 1000 == 0:
            print(f"  {desc}: {i}/{n_samples} ({100*i/n_samples:.1f}%)", flush=True)
        # tokens shape: (seq_len,) or (seq_len+1,) depending on the dataset
        # cybertron GPTDataset returns seq_len+1 tokens (input + target)
        t = np.array(tokens[:seq_len], dtype=np.uint16)
        out[i * seq_len:(i + 1) * seq_len] = t
    out.flush()
    print(f"  Done: {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


def extract_from_megatron_dataset(dataset, n_samples, out_path, seq_len, desc):
    """Extract samples from a megatron dataset in order."""
    def sample_iter():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            # sample is a dict with 'tokens' key
            if isinstance(sample, dict):
                tokens = sample['tokens']
            else:
                tokens = sample
            yield tokens
    write_bin(sample_iter(), out_path, n_samples, seq_len, desc)


def extract_val_data(out_path, n_samples=5000):
    """Extract validation data from Pile_test_5k."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    if not os.path.exists(VAL_DATA_PATH + '.bin') and not os.path.exists(VAL_DATA_PATH + '.idx'):
        print(f"WARNING: Validation data not found at {VAL_DATA_PATH}")
        print("Creating a minimal val.bin placeholder...")
        # Create a small placeholder with random tokens
        n_tokens = n_samples * SEQ_LENGTH
        data = np.zeros(n_tokens, dtype=np.uint16)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data.tofile(out_path)
        return

    d = IndexedDataset(VAL_DATA_PATH)
    print(f"Validation dataset: {len(d)} docs, extracting {n_samples} sequences of {SEQ_LENGTH}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(n_samples * SEQ_LENGTH,))

    written = 0
    doc_idx = 0
    buf = np.array([], dtype=np.uint16)

    while written < n_samples and doc_idx < len(d):
        doc = d.get(doc_idx).astype(np.uint16)
        buf = np.concatenate([buf, doc])
        doc_idx += 1

        while len(buf) >= SEQ_LENGTH and written < n_samples:
            out[written * SEQ_LENGTH:(written + 1) * SEQ_LENGTH] = buf[:SEQ_LENGTH]
            buf = buf[SEQ_LENGTH:]
            written += 1

    out.flush()
    print(f"Wrote {written} val sequences to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare cybertron data for nanogpt')
    parser.add_argument('--n_train_samples', type=int, default=479040,
                        help='Number of training samples to extract (default: full 7485-iter run)')
    parser.add_argument('--n_val_samples', type=int, default=2000,
                        help='Number of validation samples')
    parser.add_argument('--out_dir', type=str, default='data/cybertron_baseline',
                        help='Output directory')
    parser.add_argument('--skip_train', action='store_true', help='Skip training data extraction')
    parser.add_argument('--skip_val', action='store_true', help='Skip validation data extraction')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize single-process distributed (required by megatron data pipeline)
    dist = init_distributed()
    print(f"Distributed initialized: rank {dist.get_rank()}/{dist.get_world_size()}")

    # Write meta.pkl
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        with open(meta_path, 'wb') as f:
            pickle.dump({'vocab_size': VOCAB_SIZE}, f)
        print(f"Wrote {meta_path} with vocab_size={VOCAB_SIZE}")

    # Training data
    if not args.skip_train:
        train_out = os.path.join(args.out_dir, 'train.bin')
        if os.path.exists(train_out):
            print(f"Training data already exists: {train_out}")
            print("  Delete it to re-generate.")
        else:
            print(f"\nBuilding training dataset ({args.n_train_samples:,} samples)...")
            print("  This uses cached indices from:", DATA_CACHE_PATH)
            print("  If cache is missing, this may take several hours.\n")
            try:
                train_dataset = build_train_dataset(args.n_train_samples)
                extract_from_megatron_dataset(
                    train_dataset, args.n_train_samples, train_out, SEQ_LENGTH, 'train'
                )
            except Exception as e:
                print(f"\nERROR building train dataset: {e}")
                print("Hint: ensure megatron/cybertron is on PYTHONPATH and cache exists")
                raise

    # Validation data
    if not args.skip_val:
        val_out = os.path.join(args.out_dir, 'val.bin')
        if os.path.exists(val_out):
            print(f"Validation data already exists: {val_out}")
        else:
            print(f"\nExtracting validation data ({args.n_val_samples:,} samples)...")
            extract_val_data(val_out, n_samples=args.n_val_samples)

    print("\nDone! Data is ready in:", args.out_dir)
    print("Set dataset='cybertron_baseline' in your config to use it.")


if __name__ == '__main__':
    main()
