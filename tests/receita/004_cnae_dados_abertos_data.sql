BEGIN;

------------------------------------------------------------
-- DADOS FICTÍCIOS ADICIONAIS (NÃO LIMPA AS TABELAS)
--
-- Objetivos deste arquivo:
--   1) Popular eventos do mês 202606:
--      - baixas em junho/2026;
--      - aberturas em junho/2026;
--      - entrada e saída do Simples em junho/2026.
--   2) Reaproveitar competências, municípios e CNAEs já
--      cadastrados pelos arquivos 001/002/003.
--
-- Só insere empresas novas. O fato é calculado apenas para
-- as empresas deste arquivo, evitando duplicar dados antigos.
------------------------------------------------------------

------------------------------------------------------------
-- COMPETÊNCIA 202606
-- Proteção para ambientes onde o 001 ainda não tenha sido
-- atualizado com junho/2026.
------------------------------------------------------------

INSERT INTO dim_competencias
(
    competencia,
    atualizado,
    data_inclusao
)
SELECT '202606', true, DATE '2026-07-02'
WHERE NOT EXISTS
(
    SELECT 1 FROM dim_competencias d
    WHERE d.competencia = '202606'
);

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
-- ===== Empresas com BAIXA em 202606 =====
-- Entram antes de junho/2026 e encerram exatamente na competência 202606.
('40000001000101', 'Software Recife Cloud LTDA', 'Recife Cloud', 2, '6201501', '2611606', '202509', '202606', '202509', NULL, false, '2025-09-12'),
('40000002000102', 'Padaria Rio Branco Norte MEI', 'Pão Norte', 1, '1091102', '1200401', '202510', '202606', '202510', NULL, true, '2025-10-18'),
('40000003000103', 'Clínica Palmas Saúde LTDA', 'Palmas Saúde', 1, '8630503', '1721000', '202601', '202606', '202601', NULL, false, '2026-01-16'),
('40000004000104', 'Auto Center Vitória Peças LTDA', 'Vitória Peças', 2, '4530703', '3205309', '202603', '202606', '202603', NULL, false, '2026-03-22'),
('40000005000105', 'Hotel Belém Orla LTDA', 'Belém Orla', 2, '5510801', '1501402', '202604', '202606', '202604', NULL, false, '2026-04-11'),
('40000006000106', 'Fisio Joinville Movimento MEI', 'Fisio JLV', 1, '8650004', '4209102', '202605', '202606', '202605', NULL, true, '2026-05-20'),

-- ===== Entrada/Saída do Simples em 202606 =====
-- Estas empresas seguem ativas, mas geram eventos específicos de regime.
('40000007000107', 'Engenharia Campinas Projetos LTDA', 'Campinas Projetos', 2, '7112000', '3509502', '202512', NULL, '202606', NULL, false, '2025-12-14'),
('40000008000108', 'Contábil Goiânia Express MEI', 'Conta Goiânia', 1, '6920601', '5208707', '202602', NULL, '202606', NULL, true, '2026-02-09'),
('40000009000109', 'Restaurante Fortaleza Mar LTDA', 'Fortaleza Mar', 1, '5611201', '2304400', '202505', NULL, '202505', '202606', true, '2025-05-28'),
('40000010000110', 'Imóveis Curitiba Sul LTDA', 'Curitiba Sul', 1, '6810201', '4106902', '202508', NULL, '202508', '202606', true, '2025-08-26'),

-- ===== Empresas ativas sem evento especial em 202606 =====
-- Mantêm estoque de empresas ativas em junho/2026.
('40000011000111', 'Data Lake Manaus LTDA', 'Data Manaus', 3, '6311900', '1302603', '202511', NULL, '202511', NULL, false, '2025-11-07'),
('40000012000112', 'Energia Solar Campo Grande LTDA', 'Solar CG', 2, '3511501', '5002704', '202601', NULL, NULL, NULL, false, '2026-01-27'),
('40000013000113', 'Treinamento Boa Vista MEI', 'Treina RR', 1, '8599604', '1400100', '202604', NULL, '202604', NULL, true, '2026-04-23'),
('40000014000114', 'Coleta Teresina Circular LTDA', 'Circular PI', 2, '3811400', '2211001', '202605', NULL, NULL, NULL, false, '2026-05-24'),

-- ===== Aberturas em 202606 =====
-- Geram nova_empresa = true no novo último mês.
('40000015000115', 'IA Saúde Florianópolis LTDA', 'IA Saúde Floripa', 2, '6201501', '4205407', '202606', NULL, '202606', NULL, false, '2026-06-04'),
('40000016000116', 'Padaria João Pessoa Pão MEI', 'JP Pão', 1, '1091102', '2507507', '202606', NULL, '202606', NULL, true, '2026-06-11'),
('40000017000117', 'Banco Digital Salvador LTDA', 'Banco Salvador', 3, '6422100', '2927408', '202606', NULL, NULL, NULL, false, '2026-06-18'),
('40000018000118', 'Marketplace Beleza Sorocaba MEI', 'Beleza Sorocaba', 1, '9602501', '3552205', '202606', NULL, '202606', NULL, true, '2026-06-25');

------------------------------------------------------------
-- DIM EMPRESAS
-- Identidade estável: 1 linha por CNPJ
-- Só insere CNPJs que ainda não existem.
------------------------------------------------------------

INSERT INTO dim_empresas
(
    cnpj,
    data_inclusao
)
SELECT
    e.cnpj,
    e.data_inclusao
FROM tmp_empresas_base e
WHERE NOT EXISTS
(
    SELECT 1 FROM dim_empresas de
    WHERE de.cnpj = e.cnpj
);

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
-- Calculado SOMENTE para as empresas deste arquivo,
-- evitando recalcular/duplicar o fato de dados já existentes.
------------------------------------------------------------

WITH empresas_novas AS
(
    SELECT de.id
    FROM dim_empresas de
    JOIN tmp_empresas_base e
        ON e.cnpj = de.cnpj
),
snapshot_ordenado AS
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
    WHERE m.id_dim_empresa IN (SELECT id FROM empresas_novas)
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
