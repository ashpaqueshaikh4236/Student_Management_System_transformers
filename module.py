import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOOKS_AND_ARTICLES = [
    {"title": "Introduction to Machine Learning", "author": "John Doe", "topic": "Machine Learning"},
    {"title": "Deep Learning Fundamentals", "author": "Jane Smith", "topic": "Deep Learning"},
    {"title": "Data Science for Beginners", "author": "Alice Johnson", "topic": "Data Science"},
    {"title": "Advanced Algorithms", "author": "Bob Brown", "topic": "Algorithms"},
    {"title": "Modern AI Techniques", "author": "Charlie Davis", "topic": "AI"},
    {"title": "Understanding Natural Language Processing", "author": "David Wilson", "topic": "NLP"},
    {"title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", "author": "Aurélien Géron", "topic": "Machine Learning"},
    {"title": "Deep Learning", "author": "Ian Goodfellow, Yoshua Bengio, Aaron Courville", "topic": "Deep Learning"},
    {"title": "Pattern Recognition and Machine Learning", "author": "Christopher M. Bishop", "topic": "Machine Learning"},
    {"title": "Machine Learning Yearning", "author": "Andrew Ng", "topic": "Machine Learning"},
    {"title": "Python Machine Learning", "author": "Sebastian Raschka", "topic": "Machine Learning"},
    {"title": "The Hundred-Page Machine Learning Book", "author": "Andriy Burkov", "topic": "Machine Learning"},
    {"title": "Introduction to Deep Learning with Python", "author": "Francois Chollet", "topic": "Deep Learning"},
    {"title": "Natural Language Processing with Python", "author": "Steven Bird, Ewan Klein, Edward Loper", "topic": "NLP"},
    {"title": "Speech and Language Processing", "author": "Daniel Jurafsky, James H. Martin", "topic": "NLP"},
    {"title": "Deep Learning for Computer Vision", "author": "Rajalingappaa Shanmugamani", "topic": "Deep Learning"},
    {"title": "Applied Data Science with Python", "author": "Chris Albon", "topic": "Data Science"},
    {"title": "Data Science from Scratch", "author": "Joel Grus", "topic": "Data Science"},
    {"title": "Data Science Handbook", "author": "Jake VanderPlas", "topic": "Data Science"},
    {"title": "The Art of Data Science", "author": "Roger D. Peng, Elizabeth Matsui", "topic": "Data Science"},
    {"title": "Algorithm Design Manual", "author": "Steven S. Skiena", "topic": "Algorithms"},
    {"title": "Introduction to Algorithms", "author": "Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein", "topic": "Algorithms"},
    {"title": "Algorithms Unlocked", "author": "Thomas H. Cormen", "topic": "Algorithms"},
    {"title": "The Design of Everyday Things", "author": "Don Norman", "topic": "Design"},
    {"title": "The Pragmatic Programmer", "author": "Andrew Hunt, David Thomas", "topic": "Programming"},
    {"title": "Clean Code: A Handbook of Agile Software Craftsmanship", "author": "Robert C. Martin", "topic": "Programming"},
    {"title": "The Mythical Man-Month", "author": "Frederick P. Brooks Jr.", "topic": "Software Engineering"},
    {"title": "Refactoring: Improving the Design of Existing Code", "author": "Martin Fowler", "topic": "Programming"},
    {"title": "Code: The Hidden Language of Computer Hardware and Software", "author": "Charles Petzold", "topic": "Computer Science"},
    {"title": "Artificial Intelligence: A Modern Approach", "author": "Stuart Russell, Peter Norvig", "topic": "AI"},
    {"title": "Hands-On Artificial Intelligence for Cybersecurity", "author": "Rajashekara G. S. Babu", "topic": "AI"},
    {"title": "Machine Learning: A Probabilistic Perspective", "author": "Kevin P. Murphy", "topic": "Machine Learning"},
    {"title": "Building Machine Learning Powered Applications", "author": "Emmanuel Ameisen", "topic": "Machine Learning"},
    {"title": "AI Superpowers", "author": "Kai-Fu Lee", "topic": "AI"},
    {"title": "The Deep Learning Revolution", "author": "Terry Sejnowski", "topic": "Deep Learning"},
    {"title": "Deep Learning for Coders with Fastai and PyTorch", "author": "Jeremy Howard, Sylvain Gugger", "topic": "Deep Learning"},
    {"title": "Introduction to Quantum Computing", "author": "Michael A. Nielsen, Isaac L. Chuang", "topic": "Quantum Computing"},
    {"title": "Quantum Computation and Quantum Information", "author": "Michael A. Nielsen, Isaac L. Chuang", "topic": "Quantum Computing"},
    {"title": "Quantum Machine Learning", "author": "Peter Wittek", "topic": "Quantum Computing"},
    {"title": "Programming Quantum Computers", "author": "Eric R. Johnston, Nic Harrigan, and Mercedes Gimeno-Segovia", "topic": "Quantum Computing"},
    {"title": "Blockchain Basics", "author": "Daniel Drescher", "topic": "Blockchain"},
    {"title": "Mastering Bitcoin", "author": "Andreas M. Antonopoulos", "topic": "Blockchain"},
    {"title": "Blockchain Revolution", "author": "Don Tapscott, Alex Tapscott", "topic": "Blockchain"},
    {"title": "The Bitcoin Standard", "author": "Saifedean Ammous", "topic": "Blockchain"},
    {"title": "Introduction to Computer Security", "author": "Matt Bishop", "topic": "Security"},
    {"title": "Security Engineering", "author": "Ross Anderson", "topic": "Security"},
    {"title": "Computer Security: Principles and Practice", "author": "William Stallings, Lawrie Brown", "topic": "Security"},
    {"title": "Practical Cryptography for Developers", "author": "Seth James Nielson, Christopher K. Monson", "topic": "Cryptography"},
    {"title": "Cryptography and Network Security", "author": "William Stallings", "topic": "Cryptography"},
    {"title": "Compilers: Principles, Techniques, and Tools", "author": "Alfred V. Aho, Monica S. Lam, Ravi Sethi, Jeffrey D. Ullman", "topic": "Compilers"},
    {"title": "Modern Compiler Implementation in C", "author": "Andrew W. Appel", "topic": "Compilers"},
    {"title": "Computer Architecture: A Quantitative Approach", "author": "John L. Hennessy, David A. Patterson", "topic": "Computer Architecture"},
    {"title": "Structured Computer Organization", "author": "Andrew S. Tanenbaum", "topic": "Computer Architecture"},
    {"title": "Computer Systems: A Programmer's Perspective", "author": "Randal E. Bryant, David R. O'Hallaron", "topic": "Computer Systems"},
    {"title": "Operating System Concepts", "author": "Abraham Silberschatz, Henry Korth, S. Sudarshan", "topic": "Operating Systems"},
    {"title": "Modern Operating Systems", "author": "Andrew S. Tanenbaum", "topic": "Operating Systems"},
    {"title": "The C Programming Language", "author": "Brian W. Kernighan, Dennis M. Ritchie", "topic": "Programming"},
    {"title": "Programming Pearls", "author": "Jon Bentley", "topic": "Programming"},
    {"title": "Algorithms in Java", "author": "Robert Sedgewick", "topic": "Algorithms"},
    {"title": "Introduction to the Theory of Computation", "author": "Michael Sipser", "topic": "Theory of Computation"},
    {"title": "Computational Complexity: A Modern Approach", "author": "Sanjeev Arora, Boaz Barak", "topic": "Computational Complexity"},
    {"title": "Discrete Mathematics and Its Applications", "author": "Kenneth H. Rosen", "topic": "Mathematics"},
    {"title": "Introduction to Probability", "author": "D. P. Bertsekas, J. N. Tsitsiklis", "topic": "Probability"},
    {"title": "Linear Algebra and Its Applications", "author": "David C. Lay", "topic": "Mathematics"},
    {"title": "Elements of Statistical Learning", "author": "Trevor Hastie, Robert Tibshirani, Jerome Friedman", "topic": "Statistics"},
    {"title": "Bayesian Data Analysis", "author": "Andrew Gelman, John Carlin, Hal Stern, David Dunson, Aki Vehtari, Donald Rubin", "topic": "Statistics"},
    {"title": "Data Mining: Practical Machine Learning Tools and Techniques", "author": "Ian H. Witten, Eibe Frank, Mark A. Hall", "topic": "Data Science"},
    {"title": "Data Visualization: A Practical Introduction", "author": "Kieran Healy", "topic": "Data Visualization"}
]

class LLMAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
    
    def get_response(self, query):
        inputs = self.tokenizer.encode(query, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def get_recommendations(self, query):
        query_lower = query.lower()
        recommendations = []
        for item in BOOKS_AND_ARTICLES:
            if any(keyword in query_lower for keyword in item["topic"].lower().split()):
                recommendations.append(f"{item['title']} by {item['author']}")
        return recommendations if recommendations else ["No recommendations available based on your query."]

    def log_query(self, query):
        logger.info(f"Query: {query}")
