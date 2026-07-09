import wikipediaapi
import os, time

os.makedirs('data/corpus', exist_ok=True)

# You must pass a user-agent string — put your name/project
wiki = wikipediaapi.Wikipedia('SemanticSearchProject/1.0 (hrishitaanalawade@gmail.com)', 'en')

topics = [
     # AI
    "Machine learning",
    "Deep learning",
    "Large language model",
    "Transformer model",
    "Neural network",
    "Computer vision",
    "Natural language processing",
    "Reinforcement learning",
    "Generative adversarial network",
    "Decision tree",
    "Random forest",
    "Support vector machine",
    "K-means clustering",
    "Principal component analysis",
    "Bayesian network",

    # Programming
    "Python (programming language)",
    "Java (programming language)",
    "C++",
    "JavaScript",
    "TypeScript",
    "Go (programming language)",
    "Rust (programming language)",
    "C Sharp (programming language)",
    "PHP",
    "Swift (programming language)",
    "Kotlin (programming language)",

    # Databases
    "Database",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "SQLite",
    "MongoDB",
    "Redis",
    "Oracle Database",
    "Apache Cassandra",
    "Elasticsearch",

    # Cloud
    "Amazon Web Services",
    "Microsoft Azure",
    "Google Cloud Platform",
    "Docker",
    "Kubernetes",
    "Terraform",
    "Virtual machine",
    "Load balancing",
    "Microservices",
    "Serverless computing",

    # Networking
    "Computer network",
    "HTTP",
    "HTTPS",
    "TCP",
    "UDP",
    "IP address",
    "DNS",
    "Firewall",
    "Virtual private network",
    "Proxy server",

    # Operating Systems
    "Linux",
    "Windows",
    "macOS",
    "Operating system",
    "Kernel",
    "File system",
    "Process",
    "Thread",
    "Memory management",
    "Scheduling (computing)",

    # Algorithms
    "Algorithm",
    "Data structure",
    "Binary tree",
    "Graph theory",
    "Heap (data structure)",
    "Hash table",
    "Dynamic programming",
    "Greedy algorithm",
    "Breadth-first search",
    "Depth-first search",

    # Cybersecurity
    "Cryptography",
    "Encryption",
    "Authentication",
    "Authorization",
    "Cybersecurity",
    "Malware",
    "Ransomware",
    "Digital signature",
    "Public-key cryptography",
    "Hash function",

    
]
saved = 0
for topic in topics:
    try:
        page = wiki.page(topic)
        if not page.exists():
            print(f"Not found: {topic}")
            continue
        filename = topic.lower().replace(' ', '_').replace('/', '_') + '.txt'
        with open(f'data/corpus/{filename}', 'w', encoding='utf-8') as f:
            f.write(page.text)
        print(f"Saved: {filename} ({len(page.text)} chars)")
        saved += 1
        time.sleep(0.3)
    except Exception as e:
        print(f"Skipped {topic}: {e}")

print(f"\nDone — {saved} articles saved")