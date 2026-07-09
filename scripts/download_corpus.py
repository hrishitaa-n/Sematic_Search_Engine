import wikipediaapi
import os, time

os.makedirs('data/corpus', exist_ok=True)

# You must pass a user-agent string — put your name/project
wiki = wikipediaapi.Wikipedia('SemanticSearchProject/1.0 (hrishitaanalawade@gmail.com)', 'en')

topics = [
    "Machine learning", "Neural network", "Python programming language",
    "Database", "Application programming interface", "Cloud computing",
    "Computer security", "Artificial intelligence", "Data science",
    "Computer vision", "Natural language processing", "Deep learning",
    "Algorithm", "Software engineering", "Operating system",
    "Internet", "Hypertext Transfer Protocol", "Domain Name System",
    "Bitcoin", "Blockchain", "Encryption",
    "Docker", "Kubernetes", "Microservices",
    "PostgreSQL", "NoSQL", "SQL",
    "Git", "Linux", "Virtual machine",
    "Large language model", "Transformer model", "Generative adversarial network",
    "Reinforcement learning", "Computer network", "Distributed computing",
    "Load balancing", "Object-oriented programming", "Compiler",
    "World War II", "Cold War", "Space race",
    "Climate change", "Solar energy", "Electric vehicle",
    "Human genome", "CRISPR", "Vaccine",
    "Stock market", "Supply chain", "Inflation",
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