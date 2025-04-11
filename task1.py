import pandas as pd
import json
import re # For cleaning text
import string # For punctuation removal
import time
import os
import scipy.sparse # Needed for sparse matrix checks and operations

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Sentence Transformers and Torch for Dense Embeddings
try:
    from sentence_transformers import SentenceTransformer
    import torch
    print("Sentence Transformers and Torch loaded.")
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: Sentence Transformers or Torch not found. Dense embedding methods will be skipped.")
    print("Install them (`pip install sentence-transformers torch`) to enable.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Download necessary NLTK data (run only once)
print("Downloading NLTK data (if necessary)...")
nltk_packages = ['wordnet', 'stopwords', 'punkt']
for package in nltk_packages:
    try:
        if package == 'punkt':
            nltk.data.find(f'tokenizers/{package}')
        else:
             nltk.data.find(f'corpora/{package}')
        # print(f"NLTK package '{package}' already downloaded.")
    except:
        try:
           print(f"Downloading NLTK package '{package}'...")
           nltk.download(package, quiet=True)
           print(f"NLTK package '{package}' downloaded.")
        except Exception as e:
            print(f"Error downloading NLTK package '{package}': {e}")
print("NLTK check complete.")

# ## 0.1 Helper Functions (Modified and New)
print("Defining helper functions...")

# Initialize lemmatizer, stemmer and stopwords globally for efficiency
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
custom_stopwords = set([
    'claim', 'claims', 'claimed', 'method', 'system', 'device', 'apparatus', 'assembly', 'unit',
    'comprising', 'comprises', 'thereof', 'wherein', 'said', 'thereby', 'herein', 'accordance',
    'invention', 'present', 'related', 'relates', 'figure', 'fig', 'example', 'examples',
    'embodiment', 'embodiments', 'accordance', 'therein', 'associated', 'provided', 'configured',
    'includes', 'including', 'based', 'least', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'fourth', 'fifth', 'etc', 'eg', 'ie',
    'may', 'further', 'also', 'within', 'upon', 'used', 'using', 'use', 'capable', 'adapted',
    'generally', 'typically', 'respectively', 'particularly', 'preferably', 'various', 'such',
    'described', 'disclosed', 'illustrated', 'shown',
    'portion', 'member', 'element', 'surface', 'axis', 'position', 'direction', 'side', 'end', 'top', 'bottom',
    'lower', 'upper', 'inner', 'outer', 'rear', 'front', 'lateral',
    'set', 'provide', 'generate', 'control', 'controlling', 'operation', 'value', 'signal', 'process', 'data',
    'group', 'range', 'level', 'time', 'number', 'result', 'type', 'form', 'part', 'manner', 'step'
])
all_stopwords = stop_words.union(custom_stopwords)


def preprocess_text(text, use_stemming=False, use_custom_stopwords=True):
    """Enhanced preprocessing: lowercase, remove punctuation/numbers, lemmatize/stem, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # Remove single letters
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    tokens = nltk.word_tokenize(text)

    current_stopwords = all_stopwords if use_custom_stopwords else stop_words

    if use_stemming:
        processed_tokens = [stemmer.stem(word) for word in tokens if word not in current_stopwords and len(word) > 2]
    else:
        processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in current_stopwords and len(word) > 2]

    return ' '.join(processed_tokens)


def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as file: # Added encoding
            contents = json.load(file)
        return contents
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None


def create_corpus(corpus, text_type, preprocess=False, config={}):
    """
    Extracts and optionally preprocesses text data from a corpus based on the specified text type.
    Now passes config to allow conditional preprocessing.
    """
    if not corpus: # Handle case where corpus failed to load
        print(f"Warning: Attempting to create corpus from empty or None input for '{text_type}'.")
        return []

    app_ids = []
    texts = []
    cnt = 0 # count the number of documents skipped

    print(f"Creating corpus for text_type: '{text_type}'...")

    required_parts = []
    if 'title' in text_type: required_parts.append('title')
    if 'abstract' in text_type: required_parts.append('pa01')
    if 'claim1' in text_type: required_parts.append('c-en-0001')

    for doc in tqdm(corpus, desc=f"Processing {text_type}", leave=False):
        doc_id = doc.get('Application_Number', '') + doc.get('Application_Category', '')
        if not doc_id: # Skip if ID is missing
            cnt+=1
            continue
        content = doc.get('Content', {})
        if not content: # Skip if content is missing
             cnt += 1
             continue

        doc_text_parts = []
        missing_part = False

        # Simplified collection logic using a mapping
        part_map = {
            'title': ['title'],
            'abstract': ['pa01'],
            'claim1': ['c-en-0001'],
            'claims': [k for k in content if k.startswith('c-en-')],
            'description': [k for k in content if k.startswith('p')],
            'fulltext': list(content.keys())
        }

        keys_to_extract = set()
        if text_type == 'title_abstract': keys_to_extract.update(part_map['title'] + part_map['abstract'])
        elif text_type == 'title_abstract_claim1': keys_to_extract.update(part_map['title'] + part_map['abstract'] + part_map['claim1'])
        elif text_type == 'title_abstract_claims': keys_to_extract.update(part_map['title'] + part_map['abstract'] + part_map['claims'])
        elif text_type in part_map: keys_to_extract.update(part_map[text_type])
        else: print(f"Warning: Unknown text_type '{text_type}' in create_corpus.")

        # Extract text for the required keys, removing None values
        extracted_texts = [content.get(key) for key in keys_to_extract if content.get(key)]
        doc_text_parts = list(dict.fromkeys(filter(None, extracted_texts))) # Unique parts, preserving order

        # Check if required parts are missing ONLY if it's a specific type (not combo or fulltext)
        if text_type in ['title', 'abstract', 'claim1', 'claims', 'description']:
             if not doc_text_parts: # If the specific part(s) were not found
                 missing_part = True

        # Final check and processing
        if not doc_text_parts or missing_part:
            cnt += 1
        else:
            final_text = ' '.join(doc_text_parts)

            # Apply preprocessing based on config and method type
            if preprocess and config.get('method') != 'dense': # Only preprocess if requested AND method is not 'dense'
                use_stemming_flag = config.get('use_stemming', False)
                use_custom_stopwords_flag = config.get('use_custom_stopwords', True)
                final_text = preprocess_text(final_text, use_stemming=use_stemming_flag, use_custom_stopwords=use_custom_stopwords_flag)

            if not final_text or not final_text.strip():
                 cnt += 1
            else:
                texts.append(final_text)
                app_ids.append(doc_id)

    if cnt > 0:
         print(f"Number of documents skipped (missing ID/Content or required text part for '{text_type}' or empty after preprocess): {cnt}")
         final_count = len(app_ids)
         print(f"Original corpus size: {len(corpus)}. Final corpus size: {final_count}")
         if final_count == 0:
              print(f"Warning: Resulting corpus for '{text_type}' is empty!")

    corpus_data = [{'id': app_id, 'text': text} for app_id, text in zip(app_ids, texts)]
    return corpus_data


def create_tfidf_matrix(citing_texts, nonciting_texts, vectorizer=TfidfVectorizer()):
    """Creates TF-IDF matrix."""
    all_text = citing_texts + nonciting_texts
    print("Fitting TF-IDF Vectorizer...")
    vectorizer.fit(tqdm(all_text, desc="Fit TF-IDF", leave=False))
    print("Transforming Citing Texts...")
    tfidf_matrix_citing = vectorizer.transform(tqdm(citing_texts, desc="Transform Citing", leave=False))
    print("Transforming Non-Citing Texts...")
    tfidf_matrix_nonciting = vectorizer.transform(tqdm(nonciting_texts, desc="Transform Non-Citing", leave=False))
    print("Size of vocabulary:", len(vectorizer.vocabulary_))
    return tfidf_matrix_citing, tfidf_matrix_nonciting, vectorizer


class BM25Score:
    """BM25 scoring algorithm implementation."""
    def __init__(self, vectorized_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorized_docs = vectorized_docs # Should be non-citing counts

    def fit(self, vectorized_queries=None, query_ids=None, args=None):
        """Fits BM25 based on the non-citing document stats."""
        if not isinstance(self.vectorized_docs, scipy.sparse.csr_matrix):
            try:
                self.vectorized_docs = scipy.sparse.csr_matrix(self.vectorized_docs)
            except Exception as e:
                print(f"Error converting BM25 input to CSR: {e}")
                raise

        self.n_d = self.vectorized_docs.sum(axis=1).A
        self.avgdl = np.mean(self.n_d)
        if self.avgdl == 0:
            print("Warning: Average document length is zero. Setting to 1.")
            self.avgdl = 1.0

        self.n_docs = self.vectorized_docs.shape[0]
        self.nq = np.array(self.vectorized_docs.getnnz(axis=0)).reshape(1,-1)
        epsilon = 1e-9
        self.idf = np.log(((self.n_docs - self.nq + 0.5) / (self.nq + 0.5 + epsilon)) + 1.0)
        self.idf = np.maximum(self.idf, 0)
        return self

    def predict(self, vectorized_queries):
        """Calculates BM25 scores for queries against fitted documents."""
        if not isinstance(vectorized_queries, scipy.sparse.csr_matrix):
            try:
                vectorized_queries = scipy.sparse.csr_matrix(vectorized_queries)
            except Exception as e:
                print(f"Error converting BM25 query input to CSR: {e}")
                raise

        if vectorized_queries.shape[1] != self.vectorized_docs.shape[1]:
             raise ValueError(f"Query vector shape {vectorized_queries.shape} incompatible with document vector shape {self.vectorized_docs.shape}")

        idf = self.idf
        term_freq_docs = self.vectorized_docs
        term_freq_queries = vectorized_queries

        doc_len_norm_factor = self.k1 * (1 - self.b + self.b * (self.n_d / self.avgdl))
        k1_plus_1 = self.k1 + 1
        denominator = term_freq_docs.copy().astype(np.float32)

        denominator_dense = term_freq_docs.toarray() + doc_len_norm_factor
        denominator_dense[denominator_dense == 0] = 1e-9

        score_part_docs = term_freq_docs.multiply(k1_plus_1)
        score_part_docs_dense = score_part_docs.toarray() / denominator_dense

        weighted_scores = score_part_docs_dense * idf

        query_term_presence = (term_freq_queries > 0).astype(np.float32)
        final_scores = query_term_presence @ weighted_scores.T

        return final_scores


def create_bm25_matrix(citing_texts, nonciting_texts, vectorizer=CountVectorizer(), bm25_params={'k1': 1.5, 'b': 0.75}):
    """Creates BM25 similarity scores."""
    all_text = citing_texts + nonciting_texts
    print("Fitting CountVectorizer...")
    vectorizer.fit(tqdm(all_text, desc="Fit CV", leave=False))
    print("Transforming Citing Texts...")
    count_matrix_citing = vectorizer.transform(tqdm(citing_texts, desc="Transform Citing", leave=False))
    print("Transforming Non-Citing Texts...")
    count_matrix_nonciting = vectorizer.transform(tqdm(nonciting_texts, desc="Transform Non-Citing", leave=False))
    print("Size of vocabulary:", len(vectorizer.vocabulary_))
    print("Fitting BM25 model...")
    bm25 = BM25Score(count_matrix_nonciting, k1=bm25_params.get('k1', 1.5), b=bm25_params.get('b', 0.75))
    bm25.fit()
    print("Computing BM25 scores...")
    bm25_scores = bm25.predict(count_matrix_citing)
    return bm25_scores, vectorizer, bm25


def create_dense_embeddings(texts, model_name='multi-qa-mpnet-base-dot-v1', batch_size=64):
    """Generates dense embeddings for a list of texts using Sentence Transformers."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Sentence Transformers not available. Skipping dense embeddings.")
        return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for embeddings")
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        print(f"Error loading Sentence Transformer model '{model_name}': {e}")
        return None
    print(f"Generating embeddings using {model_name}...")
    try:
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
        return embeddings.detach().cpu().numpy()
    except Exception as e:
        print(f"Error during Sentence Transformer encoding: {e}")
        return None


def calculate_dense_similarity(citing_embeddings, nonciting_embeddings):
    """Calculates cosine similarity between two sets of embeddings."""
    if citing_embeddings is None or nonciting_embeddings is None:
        print("Cannot calculate dense similarity due to missing embeddings.")
        return None
    print("Calculating Dense Cosine Similarities...")
    if isinstance(citing_embeddings, torch.Tensor):
        citing_embeddings = citing_embeddings.cpu().numpy()
    if isinstance(nonciting_embeddings, torch.Tensor):
        nonciting_embeddings = nonciting_embeddings.cpu().numpy()
    try:
        similarity_scores = cosine_similarity(citing_embeddings, nonciting_embeddings)
        return similarity_scores
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return None


def get_mapping_dict(mapping_df):
    """Creates dictionary of citing ids to cited ids."""
    mapping_dict = {}
    if not isinstance(mapping_df, pd.DataFrame) or mapping_df.shape[1] < 3:
        print("Warning: mapping_df invalid in get_mapping_dict.")
        return mapping_dict
    for _, row in mapping_df.iterrows():
        try:
            key = row.iloc[0]
            value = row.iloc[2]
            if key in mapping_dict:
                mapping_dict[key].append(value)
            else:
                mapping_dict[key] = [value]
        except IndexError:
            print(f"Warning: Index error in mapping_df row: {row}")
            continue
    return mapping_dict

# --- Metrics Functions ---
def get_true_and_predicted(citing_to_cited_dict, recommendations_dict):
    true_labels = []
    predicted_labels = []
    not_in_citation_mapping = 0
    if not recommendations_dict: return [], [], 0
    for citing_id in recommendations_dict.keys():
        if citing_id in citing_to_cited_dict:
            true_labels.append(citing_to_cited_dict[citing_id])
            prediction = recommendations_dict[citing_id]
            predicted_labels.append(prediction if isinstance(prediction, list) else [])
        else:
            not_in_citation_mapping += 1
    return true_labels, predicted_labels, not_in_citation_mapping

def mean_recall_at_k(true_labels, predicted_labels, k=10):
    recalls_at_k = []
    if not true_labels or not predicted_labels: return 0.0
    for true, pred in zip(true_labels, predicted_labels):
        if not isinstance(true, (list, set)) or not isinstance(pred, list): continue
        true_set = set(true)
        if not true_set: continue
        actual_k = min(k, len(pred))
        relevant_count = sum(1 for item in pred[:actual_k] if item in true_set)
        recall = relevant_count / len(true_set)
        recalls_at_k.append(recall)
    mean_recall = sum(recalls_at_k) / len(recalls_at_k) if recalls_at_k else 0
    return mean_recall

def mean_average_precision(true_labels, predicted_labels, k=10):
    average_precisions = []
    if not true_labels or not predicted_labels: return 0.0
    for true, pred in zip(true_labels, predicted_labels):
        if not isinstance(true, (list, set)) or not isinstance(pred, list): continue
        true_set = set(true)
        if not true_set: continue
        precision_at_k = []
        relevant_count = 0
        actual_k = min(k, len(pred))
        for i, item in enumerate(pred[:actual_k]):
            if item in true_set:
                relevant_count += 1
                precision_at_k.append(relevant_count / (i + 1))
        average_precision = sum(precision_at_k) / len(true_set)
        average_precisions.append(average_precision)
    mean_average_precision_val = sum(average_precisions) / len(average_precisions) if average_precisions else 0
    return mean_average_precision_val

def mean_ranking(true_labels, predicted_labels):
    mean_ranks = []
    if not true_labels or not predicted_labels: return float('inf')
    for true, pred in zip(true_labels, predicted_labels):
        if not isinstance(true, (list, set)) or not isinstance(pred, list): continue
        if not true: continue
        ranks = []
        pred_list = list(pred) # Ensure it's indexable
        max_rank = len(pred_list) + 1
        for item in true:
            try:
                rank = pred_list.index(item) + 1
            except ValueError:
                rank = max_rank
            ranks.append(rank)
        mean_rank = sum(ranks) / len(ranks) if ranks else max_rank
        mean_ranks.append(mean_rank)
    mean_of_mean_ranks = sum(mean_ranks) / len(mean_ranks) if mean_ranks else float('inf')
    return mean_of_mean_ranks

def top_k_ranks(citing_corpus_data, nonciting_corpus_data, similarity_scores, k=10):
    """Generates top k ranks dictionary from similarity scores."""
    top_k_results = {}
    if similarity_scores is None or not citing_corpus_data or not nonciting_corpus_data:
        print("Warning: Cannot generate ranks due to missing scores or corpus data.")
        return top_k_results

    num_citing = similarity_scores.shape[0]
    num_nonciting = len(nonciting_corpus_data)

    if num_citing != len(citing_corpus_data):
         print(f"Warning: Citing scores ({num_citing}) != citing corpus ({len(citing_corpus_data)}). Adjusting...")
         num_citing = min(num_citing, len(citing_corpus_data))

    if similarity_scores.shape[1] != num_nonciting:
        print(f"Warning: Similarity score columns ({similarity_scores.shape[1]}) != non-citing docs ({num_nonciting}). Cannot rank.")
        return {}

    actual_k = min(k, num_nonciting)
    print(f"Generating top {actual_k} ranks...")
    for i in tqdm(range(num_citing), desc="Ranking", leave=False):
        try:
            citing_id = citing_corpus_data[i]['id']
            patent_scores = similarity_scores[i]
            if isinstance(patent_scores, (np.matrix, scipy.sparse.spmatrix)):
                patent_scores = patent_scores.toarray().flatten()
            elif not isinstance(patent_scores, np.ndarray):
                 patent_scores = np.array(patent_scores)

            if patent_scores.ndim != 1 or len(patent_scores) != num_nonciting:
                 print(f"Warning: Skipping citing ID {citing_id} due to score shape/length mismatch.")
                 continue

            # Argsort returns indices of the smallest values, so negate for descending order
            top_indices = np.argsort(-patent_scores)[:actual_k] # Negate scores
            top_nonciting_ids = [nonciting_corpus_data[j]['id'] for j in top_indices if j < num_nonciting]
            top_k_results[citing_id] = top_nonciting_ids
        except IndexError as e:
            print(f"Warning: Index error processing citing item {i} (ID: {citing_corpus_data[i].get('id', 'N/A')}). Skipping. Error: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error processing citing item {i} (ID: {citing_corpus_data[i].get('id', 'N/A')}): {e}. Skipping.")
    return top_k_results


def combine_rankings_rrf(rank_dict_list, k_rrf=60):
    """Combines multiple ranking dictionaries using Reciprocal Rank Fusion (RRF)."""
    print(f"Combining {len(rank_dict_list)} rankings using RRF (k={k_rrf})...")
    if not rank_dict_list or len(rank_dict_list) < 2:
        print("Warning: Need at least two ranking lists for RRF.")
        return rank_dict_list[0] if rank_dict_list else {}

    query_ids = set(rank_dict_list[0].keys())
    for r_dict in rank_dict_list[1:]:
        query_ids.intersection_update(r_dict.keys())
    if not query_ids:
        print("Warning: No common query IDs found among ranking lists for RRF.")
        return {}

    combined_scores = {query_id: {} for query_id in query_ids}
    print(f"Processing {len(query_ids)} common queries for RRF.")

    for ranks_dict in tqdm(rank_dict_list, desc="Processing Rank Lists", leave=False):
        for query_id in query_ids:
            ranked_docs = ranks_dict.get(query_id, [])
            for rank, doc_id in enumerate(ranked_docs):
                rank_score = 1.0 / (k_rrf + rank + 1)
                combined_scores[query_id][doc_id] = combined_scores[query_id].get(doc_id, 0) + rank_score

    final_rankings = {}
    for query_id, doc_scores in tqdm(combined_scores.items(), desc="Sorting RRF Results", leave=False):
        sorted_docs = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
        final_rankings[query_id] = [doc_id for doc_id, score in sorted_docs]

    return final_rankings


# --- Function to run experiments and evaluate on training data ---
# DEFINE run_experiment HERE
def run_experiment(config, json_citing_train, json_nonciting, mapping_dict, k_eval=100):
    """Runs a single experiment configuration and returns metrics."""
    start_time = time.time()
    print(f"\n--- Running Experiment: {config['name']} ---")

    if config['method'] == 'dense' and not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Skipping dense experiment as libraries are not available.")
        return None, None

    print("Creating corpora...")
    citing_corpus = create_corpus(json_citing_train, config['text_type'], preprocess=config.get('preprocess', False), config=config)
    nonciting_corpus = create_corpus(json_nonciting, config['text_type'], preprocess=config.get('preprocess', False), config=config)

    if not citing_corpus or not nonciting_corpus:
        print("Skipping experiment due to empty corpus.")
        return None, None

    citing_texts = [doc['text'] for doc in citing_corpus]
    nonciting_texts = [doc['text'] for doc in nonciting_corpus]

    similarity_scores = None
    fitted_vectorizer = None
    fitted_bm25_model = None
    nonciting_matrix_tfidf = None
    nonciting_embeddings = None
    model_details_run = {} # Initialize dictionary to hold model details

    try:
        if config['method'] == 'tfidf':
            vectorizer = TfidfVectorizer(**config.get('vectorizer_params', {}))
            tfidf_citing, tfidf_nonciting, fitted_vectorizer = create_tfidf_matrix(
                citing_texts, nonciting_texts, vectorizer
            )
            print("Calculating Cosine Similarities...")
            similarity_scores = linear_kernel(tfidf_citing, tfidf_nonciting)
            nonciting_matrix_tfidf = tfidf_nonciting

        elif config['method'] == 'bm25':
            vectorizer = CountVectorizer(**config.get('vectorizer_params', {}))
            bm25_scores, fitted_vectorizer, fitted_bm25_model = create_bm25_matrix(
                citing_texts, nonciting_texts, vectorizer, config.get('bm25_params', {})
            )
            similarity_scores = bm25_scores

        elif config['method'] == 'dense':
            print("Generating Dense Embeddings...")
            citing_embeddings = create_dense_embeddings(
                citing_texts,
                model_name=config.get('embedding_model'),
                batch_size=config.get('embedding_batch_size')
            )
            nonciting_embeddings = create_dense_embeddings(
                nonciting_texts,
                model_name=config.get('embedding_model'),
                batch_size=config.get('embedding_batch_size')
            )
            if citing_embeddings is None or nonciting_embeddings is None:
                 raise ValueError("Dense embedding generation failed.")
            similarity_scores = calculate_dense_similarity(citing_embeddings, nonciting_embeddings)
            model_details_run['nonciting_embeddings'] = nonciting_embeddings # Store needed embeddings

        else:
            print(f"Unknown method: {config['method']}")
            return None, None

        if similarity_scores is None:
            raise ValueError("Failed to compute similarity scores.")

        print(f"Shape of similarity/scores matrix: {similarity_scores.shape}")

        # Get full ranking first
        full_rank = top_k_ranks(citing_corpus, nonciting_corpus, similarity_scores, k=len(nonciting_corpus))

        # Store ranks if this config is needed for RRF
        if config['name'] == best_bm25_config_name_for_rrf:
            print(f"Storing BM25 (Best MAP/Recall) ranks for RRF from {config['name']}...")
            global best_bm25_ranks_train
            best_bm25_ranks_train = full_rank
        elif config['name'] == best_dense_config_name_for_rrf:
            print(f"Storing Dense ranks for RRF from {config['name']}...")
            global best_dense_ranks_train
            best_dense_ranks_train = full_rank
        elif config['name'] == best_mean_rank_bm25_config_name:
            print(f"Storing BM25 (Best Mean Rank) ranks for RRF from {config['name']}...")
            global best_mean_rank_bm25_ranks_train
            best_mean_rank_bm25_ranks_train = full_rank

        # Trim ranks for evaluation
        top_k_rank_eval = {qid: ranks[:k_eval] for qid, ranks in full_rank.items()}
        print("Calculating metrics...")
        true_labels, predicted_labels, not_in_mapping = get_true_and_predicted(mapping_dict, top_k_rank_eval)

        if not predicted_labels:
            print("No predictions generated for metric calculation.")
            metrics = {'recall@10': 0,'recall@20': 0,'recall@50': 0,'recall@100': 0, 'map@100': 0, 'mean_rank': float('inf'), 'num_measured': 0, 'not_in_mapping': not_in_mapping}
        else:
            metrics = {
                'recall@10': mean_recall_at_k(true_labels, predicted_labels, k=10),
                'recall@20': mean_recall_at_k(true_labels, predicted_labels, k=20),
                'recall@50': mean_recall_at_k(true_labels, predicted_labels, k=50),
                'recall@100': mean_recall_at_k(true_labels, predicted_labels, k=100),
                'map@100': mean_average_precision(true_labels, predicted_labels, k=100),
                'mean_rank': mean_ranking(true_labels, predicted_labels),
                'num_measured': len(predicted_labels), 'not_in_mapping': not_in_mapping
            }

        print(f"Recall@10: {metrics['recall@10']:.4f}")
        print(f"Recall@100: {metrics['recall@100']:.4f}")
        print(f"MAP@100: {metrics['map@100']:.4f}")
        print(f"Mean Rank: {metrics['mean_rank']:.4f}")

        # Populate model_details_run consistently
        model_details_run.update({
            'vectorizer': fitted_vectorizer, # None for dense
            'bm25_model': fitted_bm25_model, # None for tfidf/dense
            'nonciting_corpus': nonciting_corpus,
            'nonciting_matrix': nonciting_matrix_tfidf, # None for bm25/dense
            'nonciting_embeddings': model_details_run.get('nonciting_embeddings', None) # Added if dense
        })

    except Exception as e:
        print(f"Error during experiment '{config['name']}': {e}")
        import traceback
        traceback.print_exc()
        return None, None

    end_time = time.time()
    print(f"Experiment '{config['name']}' completed in {end_time - start_time:.2f} seconds.")

    return metrics, model_details_run


# # 1.0 Load Datasets
print("\nLoading datasets...")
# --- Define paths ---
DATA_DIR = "./datasets"
content_path = os.path.join(DATA_DIR, "Content_JSONs")
citation_path = os.path.join(DATA_DIR, "Citation_JSONs")
path_citing_train = os.path.join(content_path, "Citing_2020_Cleaned_Content_12k/Citing_Train_Test/citing_TRAIN.json")
path_citing_test = os.path.join(content_path, "Citing_2020_Cleaned_Content_12k/Citing_Train_Test/citing_TEST.json")
path_nonciting = os.path.join(content_path, "Cited_2020_Uncited_2010-2019_Cleaned_Content_22k/CLEANED_CONTENT_DATASET_cited_patents_by_2020_uncited_2010-2019.json")
path_citations = os.path.join(citation_path, "Citation_Train.json")

# --- Load data ---
json_citing_train = load_json_data(path_citing_train)
json_citing_test = load_json_data(path_citing_test)
json_nonciting = load_json_data(path_nonciting)
json_citing_to_cited = load_json_data(path_citations)

if not all([json_citing_train, json_citing_test, json_nonciting, json_citing_to_cited]):
    print("\nCritical Error: One or more dataset files failed to load. Please check paths. Exiting.")
    exit()

print("\nDatasets loaded successfully.")
print(f"Citing Train: {len(json_citing_train)}")
print(f"Citing Test: {len(json_citing_test)}")
print(f"Non-Citing Pool: {len(json_nonciting)}")
print(f"Training Citations Raw Pairs: {len(json_citing_to_cited)}")

mapping_dataset_df = pd.DataFrame(json_citing_to_cited)
mapping_dict = get_mapping_dict(mapping_dataset_df)
print(f"Training Citations Dict (Unique Citing Patents): {len(mapping_dict)}")

# # 2.0 Experiments Setup
print("\nSetting up experiments...")

# --- Define configurations ---
# Define base names for RRF components FIRST
best_bm25_config_name_for_rrf = 'T+A+Claims BM25 (Pre, ngram=1, k1=2.0, b=0.9)'
best_dense_config_name_for_rrf = 'Dense (multi-qa-mpnet, T+A+Claims)'
best_mean_rank_bm25_config_name = 'T+A+Claims BM25 (Pre, ngram=1, k1=2.5, b=0.8)'

configs = [
    # {'name': 'Title BM25', 'method': 'bm25', 'text_type': 'title', 'preprocess': False, 'vectorizer_params': {'stop_words': 'english', 'max_features': 10000}, 'bm25_params': {'k1': 1.5, 'b': 0.75}},
    # {'name': 'Claim1 BM25', 'method': 'bm25', 'text_type': 'claim1', 'preprocess': False, 'vectorizer_params': {'stop_words': 'english', 'max_features': 10000}, 'bm25_params': {'k1': 1.5, 'b': 0.75}},

    # {'name': best_bm25_config_name_for_rrf,
    #  'method': 'bm25', 'text_type': 'title_abstract_claims',
    #  'preprocess': True, 'use_stemming': False, 'use_custom_stopwords': True,
    #  'vectorizer_params': {'max_features': 20000, 'ngram_range': (1, 1), 'min_df': 1},
    #  'bm25_params': {'k1': 2.0, 'b': 0.9}},
    # {'name': best_mean_rank_bm25_config_name,
    #  'method': 'bm25', 'text_type': 'title_abstract_claims',
    #  'preprocess': True, 'use_stemming': False, 'use_custom_stopwords': True,
    #  'vectorizer_params': {'max_features': 20000, 'ngram_range': (1, 1), 'min_df': 1},
    #  'bm25_params': {'k1': 2.5, 'b': 0.8}},

    {'name': best_dense_config_name_for_rrf,
     'method': 'dense', 'text_type': 'title_abstract_claims', 'preprocess': False,
     'embedding_model': 'multi-qa-mpnet-base-dot-v1', 'embedding_batch_size': 128 },
    {'name': 'Dense (PatentSBERTa, T+A+Claims)', # Keep this to compare dense models
     'method': 'dense', 'text_type': 'title_abstract_claims', 'preprocess': False,
     'embedding_model': 'AI-Growth-Lab/PatentSBERTa', 'embedding_batch_size': 64 },
]

# Filter out dense methods if library not available
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    print("\nSentence Transformers not available, removing Dense configurations.")
    configs = [c for c in configs if c['method'] != 'dense']

results = {}
best_recall_100 = -1.0
best_map_100 = -1.0
best_config_name_recall = None
best_config_name_map = None
best_model_details = {} # Details of the best SINGLE model by MAP
best_model_config = None # Config of the best SINGLE model by MAP

# Initialize rank storage
best_bm25_ranks_train = None
best_dense_ranks_train = None
best_mean_rank_bm25_ranks_train = None


# --- Run Experiments ---
print("\n=== Running Experiments on Training Data ===")
k_eval_metrics = 100

if not all([json_citing_train, json_nonciting, mapping_dict]):
     print("Cannot run experiments, datasets not loaded properly.")
else:
    for config in configs:
        metrics, model_details_run = run_experiment(config, json_citing_train, json_nonciting, mapping_dict, k_eval=k_eval_metrics)
        if metrics:
            results[config['name']] = metrics
            current_recall_100 = metrics['recall@100']
            current_map_100 = metrics['map@100']

            if current_map_100 > best_map_100:
                 best_map_100 = current_map_100
                 best_config_name_map = config['name']
                 best_model_details = model_details_run
                 best_model_config = config
                 print(f"*** New best MAP@100 model found: {best_config_name_map} ({best_map_100:.4f}) ***")

            if current_recall_100 > best_recall_100:
                 best_recall_100 = current_recall_100
                 best_config_name_recall = config['name']
                 if config['name'] == best_config_name_map:
                     best_model_details = model_details_run
                     best_model_config = config
                 print(f"*** New best Recall@100 model found: {best_config_name_recall} ({best_recall_100:.4f}) ***")
        else:
            print(f"--- Experiment {config['name']} failed or produced no results. ---")


# --- Evaluate RRF on Training Data ---
print("\n=== Evaluating Hybrid RRF Variants on Training Data ===")
rrf_results = {}
best_rrf_map = -1.0
best_rrf_config_details = {}

# Check if necessary ranks were captured
rrf_possible_best_map = best_bm25_ranks_train is not None and best_dense_ranks_train is not None
rrf_possible_best_mean_rank = best_mean_rank_bm25_ranks_train is not None and best_dense_ranks_train is not None

if not rrf_possible_best_map: print(f"Warning: Ranks missing for BM25 MAP ('{best_bm25_config_name_for_rrf}') or Dense ('{best_dense_config_name_for_rrf}'). Cannot run primary RRF.")
if not rrf_possible_best_mean_rank: print(f"Warning: Ranks missing for BM25 Mean Rank ('{best_mean_rank_bm25_config_name}') or Dense ('{best_dense_config_name_for_rrf}'). Cannot run alternative RRF.")

# Variant 1: Best MAP/Recall BM25 + Best Dense, tune RRF k
if rrf_possible_best_map:
    base_rank_list = [best_bm25_ranks_train, best_dense_ranks_train]
    bm25_map_val = results.get(best_bm25_config_name_for_rrf, {}).get('map@100', 0)
    dense_map_val = results.get(best_dense_config_name_for_rrf, {}).get('map@100', 0)
    base_component_names = f"BM25(MAP={bm25_map_val:.3f}) + Dense(MAP={dense_map_val:.3f})"

    for rrf_k_val in [10, 60, 120]:
        rrf_name = f"RRF (k={rrf_k_val}, {base_component_names})"
        print(f"\nEvaluating: {rrf_name}")
        try:
            rrf_combined_ranks = combine_rankings_rrf(base_rank_list, k_rrf=rrf_k_val)
            true_labels_rrf, predicted_labels_rrf, not_in_mapping_rrf = get_true_and_predicted(mapping_dict, rrf_combined_ranks)
            if not predicted_labels_rrf: raise ValueError("No predictions from RRF combine.")

            metrics = {
                'recall@10': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=10),
                'recall@20': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=20),
                'recall@50': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=50),
                'recall@100': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=100),
                'map@100': mean_average_precision(true_labels_rrf, predicted_labels_rrf, k=100),
                'mean_rank': mean_ranking(true_labels_rrf, predicted_labels_rrf),
                'num_measured': len(predicted_labels_rrf), 'not_in_mapping': not_in_mapping_rrf
            }
            rrf_results[rrf_name] = metrics
            print(f"  RRF Metrics: R@100={metrics['recall@100']:.4f}, MAP@100={metrics['map@100']:.4f}, MeanRank={metrics['mean_rank']:.2f}")

            if metrics['map@100'] > best_rrf_map:
                best_rrf_map = metrics['map@100']
                best_rrf_config_details = {
                    'name': rrf_name, 'k': rrf_k_val,
                    'bm25_config_name': best_bm25_config_name_for_rrf,
                    'dense_config_name': best_dense_config_name_for_rrf,
                    'metrics': metrics}
                print(f"  *** New best RRF configuration found: {rrf_name} (MAP@100: {best_rrf_map:.4f}) ***")
        except Exception as e: print(f"Error evaluating {rrf_name}: {e}")

# Variant 2: Best Mean Rank BM25 + Best Dense, k=60
if rrf_possible_best_mean_rank:
    alt_rank_list = [best_mean_rank_bm25_ranks_train, best_dense_ranks_train]
    bm25_mr_val = results.get(best_mean_rank_bm25_config_name,{}).get('mean_rank', float('inf'))
    dense_map_val = results.get(best_dense_config_name_for_rrf,{}).get('map@100',0)
    alt_component_names = f"BM25(MR={bm25_mr_val:.2f}) + Dense(MAP={dense_map_val:.3f})"
    rrf_k_val = 60
    rrf_name = f"RRF (k={rrf_k_val}, {alt_component_names})"
    print(f"\nEvaluating: {rrf_name}")
    try:
        rrf_combined_ranks = combine_rankings_rrf(alt_rank_list, k_rrf=rrf_k_val)
        true_labels_rrf, predicted_labels_rrf, not_in_mapping_rrf = get_true_and_predicted(mapping_dict, rrf_combined_ranks)
        if not predicted_labels_rrf: raise ValueError("No predictions from RRF combine.")

        metrics = {
            'recall@10': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=10),
            'recall@20': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=20),
            'recall@50': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=50),
            'recall@100': mean_recall_at_k(true_labels_rrf, predicted_labels_rrf, k=100),
            'map@100': mean_average_precision(true_labels_rrf, predicted_labels_rrf, k=100),
            'mean_rank': mean_ranking(true_labels_rrf, predicted_labels_rrf),
            'num_measured': len(predicted_labels_rrf), 'not_in_mapping': not_in_mapping_rrf
        }
        rrf_results[rrf_name] = metrics
        print(f"  RRF Metrics: R@100={metrics['recall@100']:.4f}, MAP@100={metrics['map@100']:.4f}, MeanRank={metrics['mean_rank']:.2f}")

        if metrics['map@100'] > best_rrf_map:
            best_rrf_map = metrics['map@100']
            best_rrf_config_details = {
                'name': rrf_name, 'k': rrf_k_val,
                'bm25_config_name': best_mean_rank_bm25_config_name, # Use the mean rank one
                'dense_config_name': best_dense_config_name_for_rrf,
                'metrics': metrics}
            print(f"  *** New best RRF configuration found: {rrf_name} (MAP@100: {best_rrf_map:.4f}) ***")
    except Exception as e: print(f"Error evaluating {rrf_name}: {e}")

# --- Determine final best prediction method ---
print("\n--- Determining Best Prediction Method ---")
best_method_for_prediction = None
final_prediction_config = None

# Use MAP@100 as the primary decision metric
if best_rrf_map > best_map_100 and best_rrf_config_details:
    print(f"Best method is RRF: '{best_rrf_config_details['name']}' (MAP@100: {best_rrf_map:.4f})")
    best_method_for_prediction = 'rrf'
    final_prediction_config = best_rrf_config_details # Store RRF details
elif best_config_name_map:
    print(f"Best method is Single Model: '{best_config_name_map}' (MAP@100: {best_map_100:.4f})")
    best_method_for_prediction = best_config_name_map
    final_prediction_config = best_model_config # Config dict of the best single model
    if final_prediction_config: final_prediction_config['details'] = best_model_details # Attach fitted objects
else:
    print("Warning: Could not determine best method. Check experiment results and logs.")

# Add RRF results to the main results dictionary
results.update(rrf_results)

# --- Plot Results ---
print("\n=== Experiment Results Summary ===")
if results:
    results_df = pd.DataFrame(results).T.sort_values(by='map@100', ascending=False) # Sort by MAP
    pd.set_option('display.max_rows', None)
    print(results_df)
    pd.reset_option('display.max_rows')

    plt.figure(figsize=(12, 8))
    k_values_plot = [10, 20, 50, 100]
    sorted_results_plot = sorted(results.items(), key=lambda item: item[1].get('recall@100', 0), reverse=True)
    for name, metrics_res in sorted_results_plot:
        recalls = [metrics_res.get(f'recall@{k}', 0) for k in k_values_plot]
        if any(not isinstance(r, (int, float)) for r in recalls): continue
        plt.plot(k_values_plot, recalls, label=f"{name} (MAP@100: {metrics_res.get('map@100', 0):.3f})", marker='o', linewidth=1.5, markersize=5)

    plt.xlabel('Top K')
    plt.ylabel('Recall')
    plt.title('Recall@K Comparison of Methods (Train Set)')
    plt.xticks(k_values_plot)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()
else:
    print("No results to display.")


# # 4.0 Get Test Predictions for CodaBench using Best Approach
print("\n=== Generating Test Predictions for CodaBench ===")

if not best_method_for_prediction:
     print("Error: No best method determined. Cannot generate predictions.")
     exit()

print(f"Selected approach for final prediction: {best_method_for_prediction}")

k_submission = 100
test_predictions = None
output_filename = 'prediction1.json'

if best_method_for_prediction == 'rrf':
    # --- RRF Prediction Workflow ---
    print("\nGenerating RRF predictions for test set...")
    if not final_prediction_config or 'bm25_config_name' not in final_prediction_config or 'dense_config_name' not in final_prediction_config:
         print("Error: RRF selected, but configuration details are missing.")
    else:
        config_bm25 = next((c for c in configs if c['name'] == final_prediction_config['bm25_config_name']), None)
        config_dense = next((c for c in configs if c['name'] == final_prediction_config['dense_config_name']), None)
        rrf_k_val = final_prediction_config['k']

        if not config_bm25 or not config_dense:
            print("Error: Could not find original configurations for RRF components. Cannot proceed.")
        elif not SENTENCE_TRANSFORMERS_AVAILABLE and config_dense['method'] == 'dense':
             print("Error: RRF requires dense model, but sentence-transformers is not available.")
        else:
            try:
                # A. Prepare Corpora
                print("Creating corpora for RRF test prediction...")
                citing_corpus_test_bm25 = create_corpus(json_citing_test, config_bm25['text_type'], preprocess=True, config=config_bm25)
                citing_texts_test_bm25 = [doc['text'] for doc in citing_corpus_test_bm25]
                nonciting_corpus_bm25 = create_corpus(json_nonciting, config_bm25['text_type'], preprocess=True, config=config_bm25)
                nonciting_texts_bm25 = [doc['text'] for doc in nonciting_corpus_bm25]

                citing_corpus_test_dense = create_corpus(json_citing_test, config_dense['text_type'], preprocess=False, config=config_dense)
                citing_texts_test_dense = [doc['text'] for doc in citing_corpus_test_dense]
                nonciting_corpus_dense = create_corpus(json_nonciting, config_dense['text_type'], preprocess=False, config=config_dense)
                nonciting_texts_dense = [doc['text'] for doc in nonciting_corpus_dense]

                if not all([citing_corpus_test_bm25, nonciting_corpus_bm25, citing_corpus_test_dense, nonciting_corpus_dense]):
                    raise ValueError("One or more corpora creation failed for RRF.")

                # B. Get BM25 Ranks for Test Set
                print(f"\nCalculating BM25 scores for test set (using {config_bm25['name']} settings)...")
                train_corpus_bm25 = create_corpus(json_citing_train, config_bm25['text_type'], preprocess=True, config=config_bm25)
                all_train_texts_bm25 = [d['text'] for d in train_corpus_bm25] + nonciting_texts_bm25
                bm25_vectorizer = CountVectorizer(**config_bm25['vectorizer_params'])
                bm25_vectorizer.fit(tqdm(all_train_texts_bm25, desc="Fit BM25 Vectorizer", leave=False))
                test_citing_counts = bm25_vectorizer.transform(tqdm(citing_texts_test_bm25, desc="Transform Test Citing (BM25)"))
                test_nonciting_counts = bm25_vectorizer.transform(tqdm(nonciting_texts_bm25, desc="Transform Non-Citing (BM25)"))
                bm25_model_test = BM25Score(test_nonciting_counts, **config_bm25['bm25_params'])
                bm25_model_test.fit()
                test_bm25_scores = bm25_model_test.predict(test_citing_counts)
                print(f"Shape of test BM25 scores matrix: {test_bm25_scores.shape}")
                test_bm25_ranks = top_k_ranks(citing_corpus_test_bm25, nonciting_corpus_bm25, test_bm25_scores, k=max(k_submission * 2, 500)) # Increase candidate pool size

                # C. Get Dense Ranks for Test Set
                print(f"\nCalculating Dense embeddings/similarities for test set (using {config_dense['name']} settings)...")
                test_citing_embed = create_dense_embeddings(citing_texts_test_dense, model_name=config_dense['embedding_model'], batch_size=config_dense['embedding_batch_size'])
                test_nonciting_embed = create_dense_embeddings(nonciting_texts_dense, model_name=config_dense['embedding_model'], batch_size=config_dense['embedding_batch_size'])
                if test_citing_embed is None or test_nonciting_embed is None: raise ValueError("Dense embedding failed for test.")
                test_dense_sim = calculate_dense_similarity(test_citing_embed, test_nonciting_embed)
                if test_dense_sim is None: raise ValueError("Dense similarity failed for test.")
                print(f"Shape of test Dense similarity matrix: {test_dense_sim.shape}")
                test_dense_ranks = top_k_ranks(citing_corpus_test_dense, nonciting_corpus_dense, test_dense_sim, k=max(k_submission * 2, 500)) # Increase candidate pool size

                # D. Combine Ranks using RRF with the best k
                print(f"\nCombining test rankings using RRF (k={rrf_k_val})...")
                common_test_citing_ids = set(test_bm25_ranks.keys()).intersection(test_dense_ranks.keys())
                if len(common_test_citing_ids) < len(citing_corpus_test_bm25): # Check if we lost test queries
                     print(f"Warning: Mismatch in test citing IDs between BM25 ({len(test_bm25_ranks)}) and Dense ({len(test_dense_ranks)}). Using {len(common_test_citing_ids)} common IDs.")
                rank_list_for_rrf = [
                    {qid: ranks for qid, ranks in test_bm25_ranks.items() if qid in common_test_citing_ids},
                    {qid: ranks for qid, ranks in test_dense_ranks.items() if qid in common_test_citing_ids}
                ]
                test_predictions_rrf_combined = combine_rankings_rrf(rank_list_for_rrf, k_rrf=rrf_k_val)

                # E. Trim to final k for submission
                test_predictions = {qid: ranks[:k_submission] for qid, ranks in test_predictions_rrf_combined.items()}
                print(f"Generated RRF predictions for {len(test_predictions)} test patents.")

            except Exception as e:
                print(f"An error occurred during RRF test prediction generation: {e}")
                import traceback
                traceback.print_exc()
                test_predictions = None # Ensure None on error

elif best_method_for_prediction and final_prediction_config: # Fallback to best single model
    print(f"\nGenerating predictions using best single model: {best_method_for_prediction}")
    best_config = final_prediction_config
    single_model_details = best_config.get('details', {})

    if not single_model_details:
         print(f"Error: Details (fitted models) for the best single model '{best_method_for_prediction}' are missing.")
    elif best_config['method'] == 'dense' and not SENTENCE_TRANSFORMERS_AVAILABLE:
         print("Error: Best single model is dense, but sentence-transformers not available.")
    else:
        try:
            print("Creating test citing corpus...")
            citing_corpus_test = create_corpus(json_citing_test, best_config['text_type'], preprocess=best_config.get('preprocess', False), config=best_config)
            citing_texts_test = [doc['text'] for doc in citing_corpus_test]

            # Retrieve components from the *training run* details stored in best_config
            fitted_vectorizer = single_model_details.get('vectorizer')
            fitted_bm25_model = single_model_details.get('bm25_model')
            nonciting_corpus_for_ranking = single_model_details.get('nonciting_corpus')
            nonciting_matrix_tfidf = single_model_details.get('nonciting_matrix')
            nonciting_embeddings = single_model_details.get('nonciting_embeddings')

            if not citing_corpus_test or not nonciting_corpus_for_ranking:
                print("Test citing corpus or non-citing corpus for ranking is missing/empty.")
            else:
                test_similarity_scores = None
                print(f"Applying method: {best_config['method']}")
                if best_config['method'] == 'tfidf':
                     if fitted_vectorizer and nonciting_matrix_tfidf is not None:
                         citing_matrix_test = fitted_vectorizer.transform(tqdm(citing_texts_test, desc="Transform Test Citing (TFIDF)"))
                         test_similarity_scores = linear_kernel(citing_matrix_test, nonciting_matrix_tfidf)
                     else: print("Error: Missing components for TF-IDF prediction.")
                elif best_config['method'] == 'bm25':
                     if fitted_vectorizer and fitted_bm25_model:
                         citing_matrix_test = fitted_vectorizer.transform(tqdm(citing_texts_test, desc="Transform Test Citing (BM25)"))
                         test_similarity_scores = fitted_bm25_model.predict(citing_matrix_test)
                     else: print("Error: Missing components for BM25 prediction.")
                elif best_config['method'] == 'dense':
                     if nonciting_embeddings is not None:
                         citing_embeddings_test = create_dense_embeddings(
                             citing_texts_test,
                             model_name=best_config.get('embedding_model'),
                             batch_size=best_config.get('embedding_batch_size')
                         )
                         if citing_embeddings_test is not None:
                              test_similarity_scores = calculate_dense_similarity(citing_embeddings_test, nonciting_embeddings)
                         else: print("Error generating test dense embeddings.")
                     else: print("Error: Missing non-citing embeddings for dense prediction.")

                if test_similarity_scores is not None:
                    print(f"Shape of test similarity/scores matrix: {test_similarity_scores.shape}")
                    test_predictions = top_k_ranks(citing_corpus_test, nonciting_corpus_for_ranking, test_similarity_scores, k=k_submission)
                    print(f"Generated single model predictions for {len(test_predictions)} test patents.")
                else:
                    print("Failed to compute test similarity scores for single best model.")
                    test_predictions = None
        except Exception as e:
            print(f"An error occurred during single model test prediction generation: {e}")
            import traceback
            traceback.print_exc()
            test_predictions = None
else:
    print("No best model configuration identified or details missing. Cannot generate predictions.")


# 5. Save Final Predictions to JSON
if test_predictions is not None and isinstance(test_predictions, dict) and test_predictions:
    print(f"\nSaving final predictions ({len(test_predictions)} queries) to {output_filename} using method: {best_method_for_prediction}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(test_predictions, f, indent=4)
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions: {e}")
elif test_predictions is None:
     print("No predictions were generated due to errors.")
else:
     print("Predictions dictionary is empty, not saving.")


print("\nScript finished.")