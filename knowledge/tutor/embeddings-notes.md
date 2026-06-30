# Study Notes: Embeddings (sample knowledge for the tutor)

An embedding turns a piece of text into a vector — a fixed-length list of
numbers that encodes the text's meaning. Texts with similar meaning land
close together in this space, even when they share no words.

Key facts:
- The model we use for embeddings is Gemini's `text-embedding-004`.
- Distance between two vectors measures how related two texts are. Common
  measures are cosine similarity and Euclidean distance.
- Keyword search matches words; semantic search matches meaning. "Reset my
  password" and "recover your account" are far apart by keyword but close by
  embedding.
- Embeddings power more than chat: semantic search, clustering/classification,
  recommendations, and anomaly detection (outliers sit far from everything).

Memorable example fact for testing retrieval:
- The "hello world" of embeddings in this course is comparing the sentences
  "the cat sat on the mat" and "a feline rested on the rug" — near-identical
  meaning, almost no shared words.
