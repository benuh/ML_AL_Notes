"""
Evaluation utilities for RAG systems
Measure retrieval quality and generation performance
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


class RAGEvaluator:
    """Evaluate RAG system performance"""

    def __init__(self):
        self.metrics_history = []

    # ===== RETRIEVAL METRICS =====

    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Precision@K: What fraction of retrieved documents are relevant?

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision score (0-1)
        """
        if k == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)

        true_positives = len(retrieved_at_k & relevant_set)
        precision = true_positives / k

        return precision

    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Recall@K: What fraction of relevant documents were retrieved?

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall score (0-1)
        """
        if len(relevant) == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)

        true_positives = len(retrieved_at_k & relevant_set)
        recall = true_positives / len(relevant_set)

        return recall

    def f1_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        F1@K: Harmonic mean of precision and recall

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            F1 score (0-1)
        """
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def mean_reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        MRR: Position of first relevant document

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of ground truth relevant document IDs

        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant)

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / i

        return 0.0

    def average_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Average Precision: Average of precision values at each relevant doc

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of ground truth relevant document IDs

        Returns:
            AP score (0-1)
        """
        if len(relevant) == 0:
            return 0.0

        relevant_set = set(relevant)
        precisions = []

        for k in range(1, len(retrieved) + 1):
            if retrieved[k-1] in relevant_set:
                precisions.append(self.precision_at_k(retrieved, relevant, k))

        if len(precisions) == 0:
            return 0.0

        return sum(precisions) / len(relevant)

    def ndcg_at_k(self, retrieved: List[str], relevant: List[str],
                  relevance_scores: Dict[str, float] = None, k: int = 10) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            relevance_scores: Optional dict mapping doc_id to relevance score
            k: Number of top results to consider

        Returns:
            NDCG score (0-1)
        """
        if relevance_scores is None:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance_scores = {doc_id: 1.0 for doc_id in relevant}

        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)

        # IDCG: Ideal DCG (if we had perfect ranking)
        sorted_relevant = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(sorted_relevant[:k], 1):
            idcg += rel / np.log2(i + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # ===== GENERATION METRICS =====

    def answer_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure if answer is faithful to the context (no hallucinations)

        Simple version: checks if answer phrases appear in context

        Args:
            answer: Generated answer
            context: Retrieved context used

        Returns:
            Faithfulness score (0-1)
        """
        # Simple heuristic: count what fraction of answer n-grams appear in context
        answer_tokens = answer.lower().split()
        context_tokens = context.lower().split()

        if len(answer_tokens) == 0:
            return 0.0

        # Check bigrams
        answer_bigrams = [' '.join(answer_tokens[i:i+2])
                         for i in range(len(answer_tokens)-1)]

        if len(answer_bigrams) == 0:
            return 1.0  # Single word answer

        found = sum(1 for bigram in answer_bigrams
                   if bigram in ' '.join(context_tokens))

        return found / len(answer_bigrams)

    def answer_relevance(self, question: str, answer: str,
                        embedding_model=None) -> float:
        """
        Measure if answer is relevant to the question

        Args:
            question: User's question
            answer: Generated answer
            embedding_model: Optional sentence transformer model

        Returns:
            Relevance score (0-1)
        """
        if embedding_model is None:
            # Simple word overlap
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())

            if len(q_words) == 0:
                return 0.0

            overlap = len(q_words & a_words)
            return overlap / len(q_words)
        else:
            # Semantic similarity using embeddings
            from sklearn.metrics.pairwise import cosine_similarity

            q_emb = embedding_model.encode([question])
            a_emb = embedding_model.encode([answer])

            similarity = cosine_similarity(q_emb, a_emb)[0][0]
            return float(similarity)

    def context_relevance(self, question: str, context: str,
                         embedding_model=None) -> float:
        """
        Measure if retrieved context is relevant to question

        Args:
            question: User's question
            context: Retrieved context
            embedding_model: Optional sentence transformer model

        Returns:
            Relevance score (0-1)
        """
        return self.answer_relevance(question, context, embedding_model)

    # ===== COMPOSITE EVALUATION =====

    def evaluate_rag_response(self, question: str, answer: str,
                             context: str, retrieved_ids: List[str],
                             relevant_ids: List[str],
                             embedding_model=None, k: int = 5) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation

        Args:
            question: User's question
            answer: Generated answer
            context: Retrieved context
            retrieved_ids: IDs of retrieved documents
            relevant_ids: Ground truth relevant document IDs
            embedding_model: Optional for semantic metrics
            k: K for @k metrics

        Returns:
            Dictionary of metric scores
        """
        metrics = {
            # Retrieval metrics
            'precision@k': self.precision_at_k(retrieved_ids, relevant_ids, k),
            'recall@k': self.recall_at_k(retrieved_ids, relevant_ids, k),
            'f1@k': self.f1_at_k(retrieved_ids, relevant_ids, k),
            'mrr': self.mean_reciprocal_rank(retrieved_ids, relevant_ids),
            'map': self.average_precision(retrieved_ids, relevant_ids),
            'ndcg@k': self.ndcg_at_k(retrieved_ids, relevant_ids, k=k),

            # Generation metrics
            'faithfulness': self.answer_faithfulness(answer, context),
            'answer_relevance': self.answer_relevance(question, answer, embedding_model),
            'context_relevance': self.context_relevance(question, context, embedding_model)
        }

        # Store in history
        self.metrics_history.append({
            'question': question,
            'metrics': metrics
        })

        return metrics

    def summarize_metrics(self) -> Dict[str, float]:
        """
        Get average metrics across all evaluations

        Returns:
            Dictionary of average metrics
        """
        if not self.metrics_history:
            return {}

        all_metrics = [item['metrics'] for item in self.metrics_history]

        summary = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            summary[f'avg_{metric_name}'] = np.mean(values)
            summary[f'std_{metric_name}'] = np.std(values)

        return summary

    def print_evaluation_report(self, metrics: Dict[str, float]):
        """Pretty print evaluation metrics"""
        print("\n" + "="*60)
        print("RAG EVALUATION REPORT")
        print("="*60 + "\n")

        print("📊 RETRIEVAL METRICS:")
        print("-" * 60)
        print(f"  Precision@K:      {metrics['precision@k']:.3f}")
        print(f"  Recall@K:         {metrics['recall@k']:.3f}")
        print(f"  F1@K:             {metrics['f1@k']:.3f}")
        print(f"  MRR:              {metrics['mrr']:.3f}")
        print(f"  MAP:              {metrics['map']:.3f}")
        print(f"  NDCG@K:           {metrics['ndcg@k']:.3f}")

        print("\n🤖 GENERATION METRICS:")
        print("-" * 60)
        print(f"  Faithfulness:     {metrics['faithfulness']:.3f}")
        print(f"  Answer Relevance: {metrics['answer_relevance']:.3f}")
        print(f"  Context Relevance:{metrics['context_relevance']:.3f}")

        print("\n" + "="*60)

        # Interpretation
        print("\n💡 INTERPRETATION:")
        if metrics['precision@k'] < 0.5:
            print("  ⚠️  Low precision - too many irrelevant documents retrieved")
        if metrics['recall@k'] < 0.5:
            print("  ⚠️  Low recall - missing relevant documents")
        if metrics['faithfulness'] < 0.7:
            print("  ⚠️  Low faithfulness - potential hallucinations")
        if all(metrics[m] > 0.7 for m in ['precision@k', 'faithfulness', 'answer_relevance']):
            print("  ✅ Excellent RAG performance!")

        print()


class TestSetEvaluator:
    """Evaluate RAG on a test set of questions"""

    def __init__(self, rag_system, evaluator: RAGEvaluator):
        self.rag_system = rag_system
        self.evaluator = evaluator

    def create_test_set(self) -> List[Dict]:
        """
        Example test set for RAG evaluation

        Returns:
            List of test cases with questions and ground truth
        """
        test_cases = [
            {
                'question': 'What is the remote work policy?',
                'relevant_docs': ['company_policies.txt_chunk_1', 'company_policies.txt_chunk_2'],
                'expected_answer_keywords': ['remote', '3 days', 'week']
            },
            {
                'question': 'How much PTO do new employees get?',
                'relevant_docs': ['company_policies.txt_chunk_5'],
                'expected_answer_keywords': ['15 days', 'new employees']
            },
            {
                'question': 'What were the Q4 sales numbers?',
                'relevant_docs': ['sales_reports.txt_chunk_1'],
                'expected_answer_keywords': ['45.2M', 'Q4', '2023']
            }
        ]
        return test_cases

    def evaluate_test_set(self, test_cases: List[Dict],
                         embedding_model=None) -> Dict:
        """
        Evaluate RAG system on test set

        Args:
            test_cases: List of test case dictionaries
            embedding_model: Optional for semantic metrics

        Returns:
            Summary of results
        """
        results = []

        print("\n🧪 Evaluating RAG System on Test Set\n")
        print("="*70 + "\n")

        for i, test in enumerate(test_cases, 1):
            print(f"Test Case {i}/{len(test_cases)}: {test['question']}")

            # Get RAG response
            response = self.rag_system.query(test['question'], verbose=False)

            # Get retrieved doc IDs
            retrieved_ids = [doc['source'] + f"_chunk_{doc['chunk_id']}"
                           for doc in response['sources']]

            # Evaluate
            metrics = self.evaluator.evaluate_rag_response(
                question=test['question'],
                answer=response['answer'],
                context=response['context'],
                retrieved_ids=retrieved_ids,
                relevant_ids=test['relevant_docs'],
                embedding_model=embedding_model
            )

            # Check if expected keywords in answer
            answer_lower = response['answer'].lower()
            keywords_found = sum(1 for kw in test.get('expected_answer_keywords', [])
                               if kw.lower() in answer_lower)
            total_keywords = len(test.get('expected_answer_keywords', []))

            results.append({
                'question': test['question'],
                'metrics': metrics,
                'keywords_match': keywords_found / total_keywords if total_keywords > 0 else 0
            })

            print(f"  ✅ Precision@5: {metrics['precision@k']:.2f} | "
                  f"Faithfulness: {metrics['faithfulness']:.2f} | "
                  f"Keywords: {keywords_found}/{total_keywords}")
            print()

        # Summary
        summary = self.evaluator.summarize_metrics()

        print("="*70)
        print("📊 TEST SET SUMMARY")
        print("="*70 + "\n")

        for metric, value in summary.items():
            if 'avg_' in metric:
                print(f"  {metric.replace('avg_', '').upper():20s}: {value:.3f}")

        return {
            'individual_results': results,
            'summary': summary
        }


# Example usage
if __name__ == "__main__":
    # Example evaluation
    evaluator = RAGEvaluator()

    # Simulated RAG response
    metrics = evaluator.evaluate_rag_response(
        question="What is the remote work policy?",
        answer="Employees can work remotely up to 3 days per week after probation.",
        context="Remote Work Policy: Employees may work remotely up to 3 days per week...",
        retrieved_ids=['doc1', 'doc2', 'doc3'],
        relevant_ids=['doc1', 'doc2'],
        k=3
    )

    evaluator.print_evaluation_report(metrics)
