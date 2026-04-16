-- Enable MCP extension
CREATE EXTENSION IF NOT EXISTS mcp;

-- Create table for usuarios
CREATE TABLE IF NOT EXISTS usuarios (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    telefone TEXT,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

-- Create table for clientes
CREATE TABLE IF NOT EXISTS clientes (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    telefone TEXT,
    endereco TEXT,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

-- Create table for produtos
CREATE TABLE IF NOT EXISTS produtos (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL,
    descricao TEXT,
    preco REAL NOT NULL,
    categoria TEXT,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

-- Create table for vendas
CREATE TABLE IF NOT EXISTS vendas (
    id SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    produto_id INTEGER NOT NULL,
    quantidade INTEGER NOT NULL,
    valor_total REAL NOT NULL,
    data_venda DATE DEFAULT CURRENT_DATE,
    FOREIGN KEY (cliente_id) REFERENCES clientes (id),
    FOREIGN KEY (produto_id) REFERENCES produtos (id)
);

-- Create table for contas_receber
CREATE TABLE IF NOT EXISTS contas_receber (
    id SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    venda_id INTEGER,
    valor_original REAL NOT NULL,
    valor_atual REAL NOT NULL,
    data_emissao TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_vencimento TIMESTAMP NOT NULL,
    parcelas INTEGER,
    parcela_atual INTEGER,
    status CHAR(1) NOT NULL,
    forma_pagamento VARCHAR(50),
    observacoes TEXT,
    FOREIGN KEY (cliente_id) REFERENCES clientes (id),
    FOREIGN KEY (venda_id) REFERENCES vendas (id)
);
