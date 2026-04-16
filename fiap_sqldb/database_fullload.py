import os
import random
from datetime import datetime, timedelta
import psycopg2

# Database connection using DATABASE_URL environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

# Sample data for population
USERS = [
    {"nome": "Carlos Silva", "email": "carlos.silva@fiap.com.br", "telefone": "(11) 98765-4321"},
    {"nome": "Ana Oliveira", "email": "ana.oliveira@fiap.com.br", "telefone": "(11) 91234-5678"},
    {"nome": "Roberto Santos", "email": "roberto.santos@fiap.com.br", "telefone": "(11) 98765-1234"},
    {"nome": "Fernanda Costa", "email": "fernanda.costa@fiap.com.br", "telefone": "(11) 91234-8765"},
    {"nome": "Lucas Pereira", "email": "lucas.pereira@fiap.com.br", "telefone": "(11) 98765-4321"}
]

CLIENTS = [
    {"nome": "Maria Oliveira", "email": "maria.oliveira@gmail.com", "telefone": "(11) 99876-5432", "endereco": "Av. Paulista, 1000 - São Paulo/SP"},
    {"nome": "Pedro Santos", "email": "pedro.santos@yahoo.com.br", "telefone": "(11) 98765-4321", "endereco": "Rua Augusta, 500 - São Paulo/SP"},
    {"nome": "Juliana Almeida", "email": "juliana.almeida@outlook.com", "telefone": "(11) 91234-5678", "endereco": "Av. Brigadeiro Faria Lima, 2000 - São Paulo/SP"},
    {"nome": "Roberto Lima", "email": "roberto.lima@uol.com.br", "telefone": "(11) 98765-1234", "endereco": "Rua Oscar Freire, 300 - São Paulo/SP"},
    {"nome": "Camila Rocha", "email": "camila.rocha@globo.com", "telefone": "(11) 91234-8765", "endereco": "Av. das Nações Unidas, 1500 - São Paulo/SP"},
    {"nome": "André Mendes", "email": "andre.mendes@terra.com.br", "telefone": "(11) 99876-4321", "endereco": "Rua Barão de Limeira, 100 - São Paulo/SP"},
    {"nome": "Patrícia Souza", "email": "patricia.souza@bol.com.br", "telefone": "(11) 98765-5678", "endereco": "Av. Republica do Líbano, 200 - São Paulo/SP"},
    {"nome": "Diego Ferreira", "email": "diego.ferreira@ig.com.br", "telefone": "(11) 91234-4321", "endereco": "Rua da Consolação, 500 - São Paulo/SP"}
]

PRODUCTS = [
    {"nome": "Notebook Dell Inspiron", "descricao": "Notebook completo com processador Intel i7, 16GB RAM e SSD 512GB", "preco": 4999.90, "categoria": "Eletrônicos"},
    {"nome": "Smartphone Samsung Galaxy S23", "descricao": "Smartphone Android com tela de 6.2 polegadas, 128GB de armazenamento", "preco": 3599.00, "categoria": "Eletrônicos"},
    {"nome": "Tablet Apple iPad Pro", "descricao": "Tablet com processador M2, tela de 12.9 polegadas e 256GB de armazenamento", "preco": 8999.00, "categoria": "Eletrônicos"},
    {"nome": "Smart TV LG 55\" 4K", "descricao": "Televisão LED com resolução 4K, HDR e sistema webOS", "preco": 2999.90, "categoria": "Eletrônicos"},
    {"nome": "Notebook Lenovo ThinkPad X1", "descricao": "Notebook ultraleve com processador Intel i7, 8GB RAM e SSD 256GB", "preco": 5499.00, "categoria": "Eletrônicos"},
    {"nome": "Smartphone iPhone 14 Pro", "descricao": "iPhone com tela de 6.1 polegadas, chip A16 Bionic e câmera tripla", "preco": 7999.00, "categoria": "Eletrônicos"},
    {"nome": "Fone de Ouvido Sony WH-1000XM4", "descricao": "Fones sem fio com cancelamento ativo de ruído e bateria de 30 horas", "preco": 2299.90, "categoria": "Eletrônicos"},
    {"nome": "Câmera Canon EOS R5", "descricao": "Câmera mirrorless com sensor full frame e gravação em 8K", "preco": 14999.00, "categoria": "Eletrônicos"}
]

def load_users(conn):
    """Load sample users into usuarios table"""
    cur = conn.cursor()
    
    for user in USERS:
        try:
            cur.execute(
                "INSERT INTO usuarios (nome, email, telefone) VALUES (%s, %s, %s)",
                (user["nome"], user["email"], user["telefone"])
            )
            print(f"Inserted user: {user['nome']}")
        except Exception as e:
            print(f"Error inserting user {user['nome']}: {e}")
    
    conn.commit()
    cur.close()

def load_clients(conn):
    """Load sample clients into clientes table"""
    cur = conn.cursor()
    
    for client in CLIENTS:
        try:
            cur.execute(
                "INSERT INTO clientes (nome, email, telefone, endereco) VALUES (%s, %s, %s, %s)",
                (client["nome"], client["email"], client["telefone"], client["endereco"])
            )
            print(f"Inserted client: {client['nome']}")
        except Exception as e:
            print(f"Error inserting client {client['nome']}: {e}")
    
    conn.commit()
    cur.close()

def load_products(conn):
    """Load sample products into produtos table"""
    cur = conn.cursor()
    
    for product in PRODUCTS:
        try:
            cur.execute(
                "INSERT INTO produtos (nome, descricao, preco, categoria) VALUES (%s, %s, %s, %s)",
                (product["nome"], product["descricao"], product["preco"], product["categoria"])
            )
            print(f"Inserted product: {product['nome']}")
        except Exception as e:
            print(f"Error inserting product {product['nome']}: {e}")
    
    conn.commit()
    cur.close()

def load_vendas(conn):
    """Load sample sales into vendas table"""
    cur = conn.cursor()
    
    # Get all client and product IDs
    cur.execute("SELECT id FROM clientes")
    client_ids = [row[0] for row in cur.fetchall()]
    
    cur.execute("SELECT id FROM produtos")
    product_ids = [row[0] for row in cur.fetchall()]
    
    # Generate 20 sample sales
    for i in range(20):
        cliente_id = random.choice(client_ids)
        produto_id = random.choice(product_ids)
        
        # Get the product price to calculate total value
        cur.execute("SELECT preco FROM produtos WHERE id = %s", (produto_id,))
        preco = cur.fetchone()[0]
        
        quantidade = random.randint(1, 5)
        valor_total = round(preco * quantidade, 2)
        
        try:
            cur.execute(
                "INSERT INTO vendas (cliente_id, produto_id, quantidade, valor_total) VALUES (%s, %s, %s, %s)",
                (cliente_id, produto_id, quantidade, valor_total)
            )
            print(f"Inserted sale for client {cliente_id} and product {produto_id}")
        except Exception as e:
            print(f"Error inserting sale: {e}")
    
    conn.commit()
    cur.close()

def load_contas_receber(conn):
    """Load sample receivable accounts into contas_receber table"""
    cur = conn.cursor()
    
    # Get all sales IDs
    cur.execute("SELECT id FROM vendas")
    venda_ids = [row[0] for row in cur.fetchall()]
    
    # Get all client IDs
    cur.execute("SELECT id FROM clientes")
    cliente_ids = [row[0] for row in cur.fetchall()]
    
    # Generate 15 sample receivable accounts
    for i in range(15):
        cliente_id = random.choice(cliente_ids)
        
        # Randomly assign a sale ID (some may not have one)
        venda_id = random.choice(venda_ids) if random.random() > 0.3 else None
        
        valor_original = round(random.uniform(100, 10000), 2)
        valor_atual = round(valor_original * random.uniform(0.8, 1.0), 2)
        
        # Generate a date in the past (up to 90 days ago) or future (up to 30 days ahead)
        days_offset = random.randint(-90, 30)
        data_emissao = datetime.now() + timedelta(days=days_offset)
        data_vencimento = data_emissao + timedelta(days=random.randint(15, 60))
        
        parcelas = random.choice([1, 3, 6, 12])
        parcela_atual = random.randint(1, parcelas) if parcelas > 1 else 1
        
        status = random.choice(['A', 'P', 'C'])  # A=Aberto, P=Pago, C=Cancelado
        forma_pagamento = random.choice(['Cartão de Crédito', 'Boleto Bancário', 'Débito Online', 'PIX'])
        
        try:
            cur.execute(
                "INSERT INTO contas_receber (cliente_id, venda_id, valor_original, valor_atual, data_emissao, data_vencimento, parcelas, parcela_atual, status, forma_pagamento) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (cliente_id, venda_id, valor_original, valor_atual, data_emissao, data_vencimento, parcelas, parcela_atual, status, forma_pagamento)
            )
            print(f"Inserted receivable account for client {cliente_id}")
        except Exception as e:
            print(f"Error inserting receivable account: {e}")
    
    conn.commit()
    cur.close()

def main():
    """Main function to load all data"""
    try:
        conn = get_db_connection()
        
        print("Loading sample data into database...")
        
        # Load data in order to respect foreign key constraints
        load_users(conn)
        load_clients(conn)
        load_products(conn)
        load_vendas(conn)
        load_contas_receber(conn)
        
        print("\nAll sample data loaded successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()