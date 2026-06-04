BEGIN;

------------------------------------------------------------
-- LIMPEZA DOS DADOS FAKE
------------------------------------------------------------

TRUNCATE TABLE fat_empresas_mensal RESTART IDENTITY CASCADE;
TRUNCATE TABLE dim_empresas_mensal RESTART IDENTITY CASCADE;
TRUNCATE TABLE dim_empresas RESTART IDENTITY CASCADE;
TRUNCATE TABLE dim_cnaes RESTART IDENTITY CASCADE;
TRUNCATE TABLE dim_cnaes_segmento RESTART IDENTITY CASCADE;
TRUNCATE TABLE dim_municipios RESTART IDENTITY CASCADE;
TRUNCATE TABLE dim_competencias RESTART IDENTITY CASCADE;

------------------------------------------------------------
-- COMPETÊNCIAS
-- Formato: YYYYMM
------------------------------------------------------------

INSERT INTO dim_competencias
(
    competencia,
    atualizado,
    data_inclusao
)
VALUES
('202505', true, '2025-06-12'),
('202506', true, '2025-07-15'),
('202507', true, '2025-08-08'),
('202508', true, '2025-09-22'),
('202509', true, '2025-10-05'),
('202510', true, '2025-11-18'),
('202511', true, '2025-12-03'),
('202512', true, '2026-01-14'),
('202601', true, '2026-02-27'),
('202602', true, '2026-03-09'),
('202603', true, '2026-04-16'),
('202604', true, '2026-05-21'),
('202605', true, '2026-06-02');

------------------------------------------------------------
-- SEGMENTOS CNAE - SEÇÃO OFICIAL
------------------------------------------------------------

INSERT INTO dim_cnaes_segmento
(
    secao,
    denominacao,
    data_inclusao
)
VALUES
('A', 'Agricultura, pecuária, produção florestal, pesca e aquicultura', '2018-03-12'),
('D', 'Eletricidade e gás', '2019-07-25'),
('F', 'Construção', '2020-01-08'),
('G', 'Comércio; reparação de veículos automotores e motocicletas', '2017-11-30'),
('H', 'Transporte, armazenagem e correio', '2018-09-04'),
('I', 'Alojamento e alimentação', '2021-02-17'),
('J', 'Informação e comunicação', '2019-05-22'),
('P', 'Educação', '2020-10-11'),
('Q', 'Saúde humana e serviços sociais', '2018-06-19'),
('S', 'Outras atividades de serviços', '2022-04-03');

------------------------------------------------------------
-- CNAES
------------------------------------------------------------

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '6201501', 'Desenvolvimento de programas de computador sob encomenda', id, DATE '2020-08-14'
FROM dim_cnaes_segmento WHERE secao = 'J';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '6311900', 'Tratamento de dados e provedores de serviços de aplicação', id, DATE '2021-03-27'
FROM dim_cnaes_segmento WHERE secao = 'J';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '8630503', 'Atividade médica ambulatorial restrita a consultas', id, DATE '2019-11-09'
FROM dim_cnaes_segmento WHERE secao = 'Q';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '8650004', 'Atividades de fisioterapia', id, DATE '2022-05-16'
FROM dim_cnaes_segmento WHERE secao = 'Q';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '5611201', 'Restaurantes e similares', id, DATE '2018-07-21'
FROM dim_cnaes_segmento WHERE secao = 'I';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '4721102', 'Padaria e confeitaria com predominância de revenda', id, DATE '2019-12-04'
FROM dim_cnaes_segmento WHERE secao = 'G';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '8599604', 'Treinamento em desenvolvimento profissional e gerencial', id, DATE '2020-04-18'
FROM dim_cnaes_segmento WHERE secao = 'P';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '4120400', 'Construção de edifícios', id, DATE '2017-09-25'
FROM dim_cnaes_segmento WHERE secao = 'F';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '9602501', 'Cabeleireiros, manicure e pedicure', id, DATE '2021-10-30'
FROM dim_cnaes_segmento WHERE secao = 'S';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '4930202', 'Transporte rodoviário de carga', id, DATE '2018-02-13'
FROM dim_cnaes_segmento WHERE secao = 'H';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '0151201', 'Criação de bovinos para corte', id, DATE '2020-06-07'
FROM dim_cnaes_segmento WHERE secao = 'A';

INSERT INTO dim_cnaes
(
    codigo,
    nome,
    id_dim_cnae_segmento,
    data_inclusao
)
SELECT '3511501', 'Geração de energia elétrica', id, DATE '2019-08-23'
FROM dim_cnaes_segmento WHERE secao = 'D';

------------------------------------------------------------
-- MUNICÍPIOS
------------------------------------------------------------

INSERT INTO dim_municipios
(
    codigo,
    nome,
    uf,
    capital,
    data_inclusao
)
VALUES
('3550308', 'São Paulo', 'SP', true, '2015-02-10'),
('3304557', 'Rio de Janeiro', 'RJ', true, '2015-05-23'),
('3106200', 'Belo Horizonte', 'MG', true, '2015-03-22'),
('4106902', 'Curitiba', 'PR', true, '2016-01-18'),
('2927408', 'Salvador', 'BA', true, '2016-04-05'),
('2304400', 'Fortaleza', 'CE', true, '2017-07-12'),

('3509502', 'Campinas', 'SP', false, '2017-09-29'),
('3549904', 'São José dos Campos', 'SP', false, '2018-05-14'),
('3170206', 'Uberlândia', 'MG', false, '2018-11-08'),
('3304904', 'Volta Redonda', 'RJ', false, '2019-03-27'),
('4115200', 'Maringá', 'PR', false, '2019-08-19'),
('2910800', 'Feira de Santana', 'BA', false, '2020-06-30'),
('2312908', 'Sobral', 'CE', false, '2020-12-15'),
('5103403', 'Cuiabá', 'MT', true, '2021-04-21'),
('5208707', 'Goiânia', 'GO', true, '2021-09-03'),
('5002704', 'Campo Grande', 'MS', true, '2022-02-26');

------------------------------------------------------------
-- BASE TEMPORÁRIA FICTÍCIA
------------------------------------------------------------

CREATE TEMP TABLE tmp_empresas_base
(
    cnpj varchar(14) NOT NULL,
    razao_social varchar(200) NOT NULL,
    nome_fantasia varchar(100) NULL,
    porte smallint NOT NULL,
    codigo_cnae varchar(20) NOT NULL,
    codigo_municipio varchar(20) NOT NULL,
    competencia_entrada varchar(6) NOT NULL,
    competencia_baixa varchar(6) NULL,
    competencia_entrada_simples varchar(6) NULL,
    competencia_saida_simples varchar(6) NULL,
    mei boolean NOT NULL,
    data_inclusao date NOT NULL
) ON COMMIT DROP;

INSERT INTO tmp_empresas_base
(
    cnpj,
    razao_social,
    nome_fantasia,
    porte,
    codigo_cnae,
    codigo_municipio,
    competencia_entrada,
    competencia_baixa,
    competencia_entrada_simples,
    competencia_saida_simples,
    mei,
    data_inclusao
)
VALUES
-- Existentes em 05/2025
('10000001000101', 'Alpha Sistemas Digitais LTDA', 'Alpha Sistemas', 3, '6201501', '3550308', '202505', NULL, '202505', NULL, false, '2025-05-12'),
('10000002000102', 'Beta Dados e Inteligência LTDA', 'Beta Dados', 3, '6311900', '3509502', '202505', NULL, '202505', NULL, false, '2025-05-18'),
('10000003000103', 'Clínica Vida Plena LTDA', 'Vida Plena', 2, '8630503', '3304557', '202505', NULL, NULL, NULL, false, '2025-05-03'),
('10000004000104', 'Padaria Pão da Vila LTDA', 'Pão da Vila', 1, '4721102', '3106200', '202505', NULL, '202505', NULL, true, '2025-05-25'),
('10000005000105', 'Restaurante Sabor Central LTDA', 'Sabor Central', 1, '5611201', '4106902', '202505', '202604', '202505', '202603', true, '2025-05-09'),
('10000006000106', 'Construtora Pilar Forte LTDA', 'Pilar Forte', 3, '4120400', '2927408', '202505', NULL, NULL, NULL, false, '2025-05-21'),
('10000007000107', 'Logística Rota Azul LTDA', 'Rota Azul', 2, '4930202', '2304400', '202505', NULL, '202505', NULL, false, '2025-05-14'),
('10000008000108', 'Agropecuária Campo Norte LTDA', 'Campo Norte', 2, '0151201', '5103403', '202505', NULL, NULL, NULL, false, '2025-05-07'),

-- Entradas em 2025
('10000009000109', 'Studio Beleza Viva LTDA', 'Beleza Viva', 1, '9602501', '3509502', '202506', NULL, '202506', NULL, true, '2025-06-04'),
('10000010000110', 'Dev Interior Soluções LTDA', 'Dev Interior', 2, '6201501', '3549904', '202506', NULL, '202506', NULL, false, '2025-06-19'),
('10000011000111', 'Educa Mais Treinamentos LTDA', 'Educa Mais', 1, '8599604', '3170206', '202507', NULL, '202507', NULL, true, '2025-07-08'),
('10000012000112', 'Sabor da Bahia Restaurante LTDA', 'Sabor Bahia', 1, '5611201', '2910800', '202507', NULL, '202507', NULL, true, '2025-07-22'),
('10000013000113', 'Fisio Movimento LTDA', 'Fisio Movimento', 1, '8650004', '3304904', '202508', NULL, '202508', NULL, true, '2025-08-15'),
('10000014000114', 'Solaris Energia LTDA', 'Solaris Energia', 2, '3511501', '4115200', '202508', NULL, NULL, NULL, false, '2025-08-27'),
('10000015000115', 'Marmitaria Bom Prato LTDA', 'Bom Prato', 1, '5611201', '3550308', '202509', '202602', '202509', NULL, true, '2025-09-03'),
('10000016000116', 'Data Lake Brasil LTDA', 'Data Lake Brasil', 3, '6311900', '3509502', '202509', NULL, '202509', NULL, false, '2025-09-18'),
('10000017000117', 'Casa e Obra Serviços LTDA', 'Casa e Obra', 1, '4120400', '5208707', '202510', NULL, '202510', NULL, true, '2025-10-11'),
('10000018000118', 'Beauty Express Sobral LTDA', 'Beauty Express', 1, '9602501', '2312908', '202510', NULL, '202510', NULL, true, '2025-10-24'),
('10000019000119', 'Tech Saúde Digital LTDA', 'Tech Saúde', 2, '6201501', '3304557', '202511', NULL, '202511', NULL, false, '2025-11-06'),
('10000020000120', 'Cargas Centro Oeste LTDA', 'Cargas CO', 2, '4930202', '5002704', '202511', NULL, NULL, NULL, false, '2025-11-20'),
('10000021000121', 'Cursos Pro Online LTDA', 'Cursos Pro', 1, '8599604', '3550308', '202512', NULL, '202512', NULL, true, '2025-12-09'),
('10000022000122', 'Agro Inova Rural LTDA', 'Agro Inova', 2, '0151201', '3170206', '202512', NULL, NULL, NULL, false, '2025-12-23'),

-- Entradas em 2026
('10000023000123', 'IA Aplicada Consultoria LTDA', 'IA Aplicada', 2, '6201501', '3549904', '202601', NULL, '202601', NULL, false, '2026-01-13'),
('10000024000124', 'Nuvem Fiscal Dados LTDA', 'Nuvem Fiscal', 2, '6311900', '3509502', '202601', NULL, '202601', NULL, false, '2026-01-29'),
('10000025000125', 'Energia Limpa Interior LTDA', 'Energia Limpa', 2, '3511501', '4115200', '202602', NULL, NULL, NULL, false, '2026-02-04'),
('10000026000126', 'Clínica Popular Horizonte LTDA', 'Clínica Horizonte', 1, '8630503', '3106200', '202602', NULL, '202602', NULL, true, '2026-02-17'),
('10000027000127', 'Manicure Digital MEI', 'Manicure Digital', 1, '9602501', '2304400', '202603', NULL, '202603', NULL, true, '2026-03-10'),
('10000028000128', 'Analytics Interior LTDA', 'Analytics Interior', 2, '6311900', '3170206', '202603', NULL, '202603', NULL, false, '2026-03-25'),
('10000029000129', 'Software Vale Tech LTDA', 'Vale Tech', 2, '6201501', '3549904', '202604', NULL, '202604', NULL, false, '2026-04-08'),
('10000030000130', 'Energia Solar Nordeste LTDA', 'Solar Nordeste', 2, '3511501', '2910800', '202604', NULL, NULL, NULL, false, '2026-04-19'),
('10000031000131', 'Treina AI Educação LTDA', 'Treina AI', 1, '8599604', '3550308', '202604', NULL, '202604', NULL, true, '2026-04-30'),
('10000032000132', 'Fisio Casa Interior LTDA', 'Fisio Casa', 1, '8650004', '2312908', '202605', NULL, '202605', NULL, true, '2026-05-05'),
('10000033000133', 'Plataforma de Dados Médicos LTDA', 'Dados Médicos', 2, '6311900', '3304557', '202605', NULL, '202605', NULL, false, '2026-05-14'),
('10000034000134', 'Construção Modular Verde LTDA', 'Modular Verde', 2, '4120400', '4115200', '202605', NULL, '202605', NULL, false, '2026-05-21'),
('10000035000135', 'Energia Solar Paulista LTDA', 'Solar Paulista', 2, '3511501', '3509502', '202605', NULL, NULL, NULL, false, '2026-05-27'),
('10000036000136', 'Marketplace de Beleza MEI', 'Beauty Market', 1, '9602501', '3550308', '202605', NULL, '202605', NULL, true, '2026-05-30');

------------------------------------------------------------
-- DIM EMPRESAS
-- Identidade estável: 1 linha por CNPJ
------------------------------------------------------------

INSERT INTO dim_empresas
(
    cnpj,
    data_inclusao
)
SELECT
    cnpj,
    data_inclusao
FROM tmp_empresas_base;

------------------------------------------------------------
-- SNAPSHOT MENSAL
------------------------------------------------------------

INSERT INTO dim_empresas_mensal
(
    cnpj,
    razao_social,
    nome_fantasia,
    porte,
    situacao,
    id_dim_empresa,
    id_dim_cnae,
    id_dim_municipio,
    id_dim_competencia,
    simples,
    mei,
    data_inclusao
)
SELECT
    e.cnpj,
    e.razao_social,
    e.nome_fantasia,
    e.porte,

    CASE
        WHEN e.competencia_baixa IS NOT NULL
         AND c.competencia >= e.competencia_baixa
            THEN 8
        ELSE 2
    END AS situacao,

    de.id AS id_dim_empresa,
    dc.id AS id_dim_cnae,
    dm.id AS id_dim_municipio,
    c.id AS id_dim_competencia,

    CASE
        WHEN e.competencia_entrada_simples IS NOT NULL
         AND c.competencia >= e.competencia_entrada_simples
         AND (
                e.competencia_saida_simples IS NULL
                OR c.competencia < e.competencia_saida_simples
             )
            THEN true
        ELSE false
    END AS simples,

    e.mei,

    MAKE_DATE(
        SUBSTRING(c.competencia FROM 1 FOR 4)::int,
        SUBSTRING(c.competencia FROM 5 FOR 2)::int,
        (((de.id::int * 11) + (dc.id * 7) + (dm.id * 3) + (c.id * 5)) % 27 + 1)
    ) AS data_inclusao
FROM tmp_empresas_base e
JOIN dim_competencias c
    ON c.competencia >= e.competencia_entrada
JOIN dim_empresas de
    ON de.cnpj = e.cnpj
JOIN dim_cnaes dc
    ON dc.codigo = e.codigo_cnae
JOIN dim_municipios dm
    ON dm.codigo = e.codigo_municipio;

------------------------------------------------------------
-- FATO EMPRESAS MENSAL
------------------------------------------------------------

WITH snapshot_ordenado AS
(
    SELECT
        m.id_dim_competencia,
        c.competencia,
        m.id_dim_empresa,
        m.situacao,
        m.simples,
        m.data_inclusao,

        LAG(m.situacao) OVER
        (
            PARTITION BY m.id_dim_empresa
            ORDER BY c.competencia
        ) AS situacao_anterior,

        LAG(m.simples) OVER
        (
            PARTITION BY m.id_dim_empresa
            ORDER BY c.competencia
        ) AS simples_anterior,

        ROW_NUMBER() OVER
        (
            PARTITION BY m.id_dim_empresa
            ORDER BY c.competencia
        ) AS ordem_empresa
    FROM dim_empresas_mensal m
    JOIN dim_competencias c
        ON c.id = m.id_dim_competencia
)
INSERT INTO fat_empresas_mensal
(
    id_dim_competencia,
    id_dim_empresa,
    nova_empresa,
    baixada,
    mudou_situacao,
    entrou_simples,
    saiu_simples,
    data_inclusao
)
SELECT
    id_dim_competencia,
    id_dim_empresa,

    CASE
        WHEN ordem_empresa = 1 THEN true
        ELSE false
    END AS nova_empresa,

    CASE
        WHEN situacao_anterior IS NOT NULL
         AND situacao_anterior <> 8
         AND situacao = 8
            THEN true
        ELSE false
    END AS baixada,

    CASE
        WHEN situacao_anterior IS NOT NULL
         AND situacao_anterior <> situacao
            THEN true
        ELSE false
    END AS mudou_situacao,

    CASE
        WHEN simples_anterior IS NOT NULL
         AND simples_anterior = false
         AND simples = true
            THEN true
        ELSE false
    END AS entrou_simples,

    CASE
        WHEN simples_anterior IS NOT NULL
         AND simples_anterior = true
         AND simples = false
            THEN true
        ELSE false
    END AS saiu_simples,

    data_inclusao
FROM snapshot_ordenado;

COMMIT;