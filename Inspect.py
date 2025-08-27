# OPTIONAL: Reload the BERTopic model and embeddings without re-training (e.g., in a new session)
# from bertopic import BERTopic
# import pickle
# topic_model = BERTopic.load("bertopic_thesis_model")          # Load the saved BERTopic model
# with open("thesis_embeddings_minilm.pkl", "rb") as f:
#     embeddings = pickle.load(f)                               # Load precomputed document embeddings
#
# # Example: Print topics for a new document (or an existing one) using the loaded model
# new_doc = "Example text of a new thesis abstract..."
# new_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([new_doc])
# new_topic, new_prob = topic_model.transform([new_doc], embeddings=new_embedding)
# print(f"New document assigned to Topic {new_topic[0]} with probability {max(new_prob[0]):.2f}")
