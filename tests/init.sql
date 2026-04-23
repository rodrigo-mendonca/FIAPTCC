-- ==========================================================
-- 1. ESTRUTURA DAS TABELAS (DDL)
-- ==========================================================

CREATE TABLE IF NOT EXISTS usuarios (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    telefone TEXT,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS clientes (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    telefone TEXT,
    endereco TEXT,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS produtos (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL,
    descricao TEXT,
    preco NUMERIC(12,2) NOT NULL,
    categoria TEXT,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS vendas (
    id SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    produto_id INTEGER NOT NULL REFERENCES produtos(id),
    quantidade INTEGER NOT NULL,
    valor_total NUMERIC(12,2) NOT NULL,
    data_venda TIMESTAMP DEFAULT (CURRENT_TIMESTAMP - (random() * interval '365 days'))
);

CREATE TABLE IF NOT EXISTS contas_pagar (
    id SERIAL PRIMARY KEY,
    fornecedor TEXT NOT NULL,
    valor NUMERIC(12,2) NOT NULL,
    data_vencimento DATE NOT NULL,
    status VARCHAR(20) CHECK (status IN ('Aberto', 'Pago', 'Atrasado')),
    categoria_despesa TEXT
);

-- ==========================================================
-- 2. POPULANDO 50 USUÁRIOS (Nomes Reais de Funcionários)
-- ==========================================================
DO $$
DECLARE 
    v_nomes TEXT[] := ARRAY['Marcelo', 'Rodrigo', 'Felipe', 'André', 'Ricardo', 'Fernando', 'Gustavo', 'Leonardo', 'Matheus', 'Eduardo', 'Juliana', 'Beatriz', 'Fernanda', 'Camila', 'Letícia'];
    v_sobrenomes TEXT[] := ARRAY['Silva', 'Santos', 'Oliveira', 'Souza', 'Rodrigues', 'Ferreira', 'Alves', 'Pereira', 'Lima', 'Gomes'];
BEGIN
    FOR i IN 1..50 LOOP
        INSERT INTO usuarios (nome, email, telefone)
        VALUES (
            v_nomes[floor(random() * 15 + 1)] || ' ' || v_sobrenomes[floor(random() * 10 + 1)] || ' ' || i,
            'colaborador' || i || '@empresa.com.br',
            '(11) 9' || floor(random() * 8999 + 1000)::text || '-' || floor(random() * 8999 + 1000)::text
        );
    END LOOP;
END $$;

-- ==========================================================
-- 3. POPULANDO 100 CLIENTES (Nomes Reais)
-- ==========================================================
DO $$
DECLARE 
    v_n TEXT[] := ARRAY['Gabriel', 'Lucas', 'Rafael', 'Thiago', 'Bruno', 'Aline', 'Patrícia', 'Vanessa', 'Bianca', 'Carolina', 'Mariana', 'Jéssica', 'Débora', 'Vitor', 'Heitor'];
    v_s TEXT[] := ARRAY['Costa', 'Ribeiro', 'Martins', 'Carvalho', 'Almeida', 'Lopes', 'Soares', 'Fernandes', 'Vieira', 'Barbosa'];
BEGIN
    FOR i IN 1..100 LOOP
        INSERT INTO clientes (nome, email, telefone, endereco)
        VALUES (
            v_n[floor(random() * 15 + 1)] || ' ' || v_s[floor(random() * 10 + 1)] || ' ' || i,
            'contato_cliente' || i || '@gmail.com',
            '(21) 9' || floor(random() * 8999 + 1000)::text || '-' || floor(random() * 8999 + 1000)::text,
            'Rua Principal, número ' || i || ', Centro'
        );
    END LOOP;
END $$;

-- ==========================================================
-- 4. POPULANDO 600 PRODUTOS (Mix de Tecnologia e Eletrónicos)
-- ==========================================================
DO $$
DECLARE
    v_marcas TEXT[] := ARRAY['Apple', 'Samsung', 'Dell', 'LG', 'Sony', 'Logitech', 'Asus', 'Lenovo', 'Xiaomi', 'Razer'];
    v_tipos TEXT[] := ARRAY['Smartphone', 'Notebook', 'Monitor', 'Teclado Mecânico', 'Mouse Gamer', 'Headset 7.1', 'Placa de Vídeo', 'SSD NVMe', 'Smartwatch', 'Tablet'];
    v_modelos TEXT[] := ARRAY['Pro', 'Max', 'Ultra', 'Plus', 'Air', 'Series X', 'Elite', 'Gaming', 'V2', 'Prime'];
BEGIN
    FOR i IN 1..600 LOOP
        INSERT INTO produtos (nome, descricao, preco, categoria)
        VALUES (
            v_marcas[floor(random() * 10 + 1)] || ' ' || v_tipos[floor(random() * 10 + 1)] || ' ' || v_modelos[floor(random() * 10 + 1)] || ' (Lote ' || i || ')',
            'Equipamento de alta performance, certificado de garantia incluso.',
            round((random() * 8000 + 100)::numeric, 2),
            (ARRAY['Informática', 'Gaming', 'Mobile', 'Periféricos'])[floor(random() * 4 + 1)]
        );
    END LOOP;
END $$;

-- ==========================================================
-- 5. POPULANDO 2000 VENDAS (Histórico de 1 Ano)
-- ==========================================================
DO $$
DECLARE
    v_cli_id INT;
    v_prod_id INT;
    v_preco NUMERIC;
    v_qtd INT;
BEGIN
    FOR i IN 1..2000 LOOP
        -- Sorteia um cliente e um produto existente
        SELECT id INTO v_cli_id FROM clientes ORDER BY random() LIMIT 1;
        SELECT id, preco INTO v_prod_id, v_preco FROM produtos ORDER BY random() LIMIT 1;
        v_qtd := floor(random() * 4 + 1);

        INSERT INTO vendas (cliente_id, produto_id, quantidade, valor_total)
        VALUES (v_cli_id, v_prod_id, v_qtd, (v_preco * v_qtd));
    END LOOP;
END $$;

-- ==========================================================
-- 6. POPULANDO 500 CONTAS A PAGAR (Despesas de Empresa)
-- ==========================================================
DO $$
DECLARE
    v_forn TEXT[] := ARRAY['Distribuidora Tech Brasil', 'Logística Express Ltda', 'Serviços de Cloud SA', 'Office Depot Fornecimentos', 'Imobiliária Horizonte'];
BEGIN
    FOR i IN 1..500 LOOP
        INSERT INTO contas_pagar (fornecedor, valor, data_vencimento, status, categoria_despesa)
        VALUES (
            v_forn[floor(random() * 5 + 1)],
            round((random() * 5000 + 200)::numeric, 2),
            CURRENT_DATE + (floor(random() * 90)::int || ' days')::interval,
            (ARRAY['Aberto', 'Pago', 'Atrasado'])[floor(random() * 3 + 1)],
            (ARRAY['Estoque', 'Aluguel', 'Marketing', 'Manutenção'])[floor(random() * 4 + 1)]
        );
    END LOOP;
END $$;