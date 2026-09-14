export interface ResultRow {
  mode: string;
  ndcg: string;
  recall: string;
  latency: string;
  share: number;
  winner?: boolean;
}

export const headlineResults: ResultRow[] = [
  { mode: "bm25", ndcg: "0.6569", recall: "0.7784", latency: "0.9", share: 30 },
  { mode: "vector (flat)", ndcg: "0.6451", recall: "0.7833", latency: "0.1", share: 30 },
  { mode: "hybrid_rrf", ndcg: "0.7016", recall: "0.8482", latency: "1.1", share: 32, winner: true },
  { mode: "hybrid_rerank", ndcg: "0.6820", recall: "0.8062", latency: "167.3", share: 8 },
];

export interface ScaleResultRow extends ResultRow {
  scale: string;
}

export const scaleResults: ScaleResultRow[] = [
  { scale: "5K", mode: "bm25", ndcg: "0.6569", recall: "0.7784", latency: "0.9", share: 30 },
  { scale: "5K", mode: "vector (flat)", ndcg: "0.6451", recall: "0.7833", latency: "0.1", share: 30 },
  { scale: "5K", mode: "hybrid_rrf", ndcg: "0.7016", recall: "0.8482", latency: "1.1", share: 33, winner: true },
  { scale: "5K", mode: "hybrid_rerank", ndcg: "0.6820", recall: "0.8062", latency: "167.3", share: 7 },
  { scale: "100K", mode: "bm25", ndcg: "0.3206", recall: "0.3793", latency: "6.5", share: 32 },
  { scale: "100K", mode: "vector (IVFPQ+RFlat)", ndcg: "0.1674", recall: "0.1971", latency: "0.05", share: 17 },
  { scale: "100K", mode: "hybrid_rrf", ndcg: "0.3232", recall: "0.3796", latency: "6.8", share: 32, winner: true },
  { scale: "100K", mode: "hybrid_rerank", ndcg: "0.3362", recall: "0.3871", latency: "187.7", share: 19 },
];

export interface StatItem {
  value: string;
  label: string;
  detail: string;
}

export const stats: StatItem[] = [
  { value: "+0.045", label: "nDCG@10 over BM25", detail: "at 5K untouched scifact, hybrid RRF wins" },
  { value: "6 GB", label: "corpus, ~5 min", detail: "first-run BEIR distractor cache" },
  { value: "0.6789", label: "BM25 pre-flight ±0.03", detail: "Anserini equivalence pinned at startup" },
  { value: "101", label: "tests passing", detail: "with a golden pre-flight anchor" },
];

export interface FeatureItem {
  icon: "sparkles" | "puzzle" | "shield" | "gauge" | "branch" | "terminal";
  title: string;
  description: string;
  detail: string;
  color: "brand" | "accent" | "ink";
}

export const features: FeatureItem[] = [
  {
    icon: "shield",
    title: "Reproducible by construction",
    description:
      "Every CSV row carries Python, OS, NumPy, faiss, sentence-transformers, torch, bm25s, OMP threads, and git SHA. A BM25 pre-flight pins the implementation to the published Anserini baseline.",
    detail: "Fail fast at 0.62. The benchmark is honest.",
    color: "brand",
  },
  {
    icon: "puzzle",
    title: "One decorator, one pipeline",
    description:
      "Adding a new encoder, reranker, or pipeline mode is a single @register_pipeline call. The registry-driven runner has no elif chains — every mode goes through the same gate.",
    detail: "From a new model to a row in benchmark.csv.",
    color: "accent",
  },
  {
    icon: "gauge",
    title: "Real latency, not estimates",
    description:
      "pytrec_eval relevance grading, perf_counter_ns() per query, 20-query warmup, single-stream. The published 20–40 ms rerank number turned out to be 5–10× too low — we updated it.",
    detail: "If a number is wrong, we change it.",
    color: "brand",
  },
  {
    icon: "sparkles",
    title: "Distractors that bite",
    description:
      "SCIDOCS + TREC-COVID + NFCorpus distractors prefixed by ID, deduped by SHA-256 and MinHash. The 100K curve is a true prefix of 1M so scale comparisons stay clean.",
    detail: "No closed-world optimism. No leakage.",
    color: "accent",
  },
  {
    icon: "branch",
    title: "Pareto frontier by default",
    description:
      "OPQ + IVFPQ + RFlat with nlist = 4·sqrt(N), nprobe sweep over 8–256, k_factor = 4 for RFlat refinement. Every scale gets its own frontier; no single-knob tuning.",
    detail: "Best speed/recall is a sweep, not a setting.",
    color: "ink",
  },
  {
    icon: "terminal",
    title: "Public API you'll actually use",
    description:
      "Bm25, Faiss.Flat, Rrf, get_reranker — composable primitives with the same retriever ABC. The benchmark is the reference workload; the library is what you ship.",
    detail: "Write the pipeline once, run it everywhere.",
    color: "brand",
  },
];

export interface PipelineStep {
  id: string;
  label: string;
  description: string;
}

export const pipelineSteps: PipelineStep[] = [
  { id: "corpus", label: "Corpus", description: "scifact 5K + 95K SCIDOCS/TREC-COVID/NFCorpus distractors" },
  { id: "embed", label: "Embed", description: "all-MiniLM-L6-v2 (384d, l2-normalized)" },
  { id: "index", label: "Index", description: "Faiss Flat / OPQ48 + IVFPQ + RFlat at scale" },
  { id: "fuse", label: "Fuse", description: "RRF (k=60) over BM25 + vector top-100" },
  { id: "rerank", label: "Rerank", description: "ms-marco-MiniLM-L-6 cross-encoder top-50 → 10" },
  { id: "score", label: "Score", description: "pytrec_eval nDCG@10, R@10, latency p95" },
];

export const audiences = [
  {
    title: "Retrieval engineers",
    description:
      "Choosing a fusion strategy for a real product. Need apples-to-apples numbers across BM25, vector, RRF, and rerank — at the scale that matters.",
    detail: "Pareto curves, not single points.",
  },
  {
    title: "ML researchers",
    description:
      "Validating a new encoder or reranker. Need a defensible reference workload that survives reviewer scrutiny — closed-world qrels, untouched scifact, reproducible manifests.",
    detail: "ReproducibilityManifest.gather() per row.",
  },
  {
    title: "OSS maintainers",
    description:
      "Want to publish numbers you can defend. Need a benchmark with pre-flight checks, golden anchors, and a methodology section that explains every knob in the YAML.",
    detail: "Fail fast. Update the number. Ship it.",
  },
];

export interface LogoItem {
  name: string;
}

export const libraries: LogoItem[] = [
  { name: "FAISS" },
  { name: "BM25s" },
  { name: "BEIR" },
  { name: "Pytrec Eval" },
  { name: "Sentence Transformers" },
  { name: "PyTorch" },
  { name: "Pydantic" },
  { name: "MinHash LSH" },
];