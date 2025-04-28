from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- 配置 (与导入脚本一致) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# 匹配最新的集合名称
COLLECTION_NAME = "wow_guides_en_chunked"
# 匹配最新的英文模型
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- 初始化 ---
# print("Initializing Qdrant client...")
# client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT) # 注释掉本地连接

# 添加 Qdrant Cloud 连接
print("Initializing Qdrant Cloud client...")
client = QdrantClient(
    url="https://048a9c04-5c3d-49fc-9e64-4c555f33c4b5.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cyryM5Bho5E4Nb7P9pPl-YsjZTImb2ijl-KpaWCEwNs",
)
print("Qdrant Cloud client initialized.")

print(f"Loading Embedding model: {EMBEDDING_MODEL_NAME}...")
try:
    # encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True) # If needed
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.")
    vector_size = encoder.get_sentence_embedding_dimension()
    print(f"Vector dimension: {vector_size}")
except Exception as e:
    print(f"!!!!!! Critical error loading Embedding model: {e} !!!!!!")
    import traceback
    traceback.print_exc()
    exit()

# --- 测试查询函数 ---
def test_query(query_text, limit=3, filter_type=None):
    """Runs a search query against Qdrant and prints the results."""
    print(f"\n--- Query: '{query_text}' ---")
    try:
        query_vector = encoder.encode(query_text).tolist()

        query_filter = None
        filter_log = ""
        if filter_type:
            filter_log = f"(filtered by type='{filter_type}')"
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value=filter_type))
                ]
            )

        print(f"Searching in collection '{COLLECTION_NAME}' {filter_log}...")
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )

        print("Search Results:")
        if search_result:
            for i, hit in enumerate(search_result):
                print(f"  Result {i+1}:")
                print(f"    ID: {hit.id}")
                print(f"    Score: {hit.score:.4f}")
                payload = hit.payload # Get the payload dictionary
                print(f"    Source: {payload.get('source_file', 'N/A')}")
                print(f"    Title: {payload.get('title', 'N/A')}")
                # Directly print the English chunk text
                print(f"    Chunk Text (English): {payload.get('chunk_text', '')[:250]}...") # Show more chars
                print("-" * 10)
        else:
            print("No relevant results found.")

    except Exception as e:
        print(f"Error during query: {e}")

# --- 执行英文测试 ---
if __name__ == "__main__":
    # English test questions
    test_query("What does Protection Warrior's Shield Slam do?")
    test_query("What are the core talents for Fury Warrior?", filter_type="class_guide")
    test_query("Where is the entrance to Ara-Kara, City of Echoes?", filter_type="dungeon_overview")
    test_query("In City of Threads, after defeating the first boss, what special task must be completed to proceed?")
    test_query("What is the name of the skill of BOSS Speaker Shadowcrown?")
    test_query("What is the DPS rotation for Fury Warrior?", limit=1)
    test_query("Tell me about the Protection Warrior spec.")
    test_query("What is the stat priority for Fury Warrior?")

    # 你可以添加更多英文测试问题
    # test_query("What are the abilities of the first boss in The Rookery?")
    # test_query("How do Mythic+ Keystones work in The War Within?")
