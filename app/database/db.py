# Demo in-memory database (no MongoDB required)

class InMemoryCollection:
    def __init__(self):
        self.data = []
    
    def insert_one(self, doc):
        self.data.append(doc)
        return {"inserted_id": len(self.data)}
    
    def find_one(self, query):
        for doc in self.data:
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None
    
    def find(self, query=None):
        if query is None:
            return self.data
        return [doc for doc in self.data if all(doc.get(k) == v for k, v in query.items())]

users_collection = InMemoryCollection()
reports_collection = InMemoryCollection()
appointments_collection = InMemoryCollection()