BEGIN;

------------------------------------------------------------
-- DADOS FICTÍCIOS ADICIONAIS (NÃO LIMPA AS TABELAS)
--
-- Objetivos deste arquivo:
--   1) Popular a métrica "Baixas (último mês)" do dashboard:
--      cria empresas que entram antes de 202605 e têm
--      competencia_baixa = 202605, gerando baixada = true na
--      última competência.
--   2) Ampliar a cobertura geográfica: adiciona empresas em
--      estados ainda não representados (AC, AL, AP, ES, PB,
--      PI, RN, RO, RR, TO).
--
-- Reaproveita competências, segmentos e CNAEs já cadastrados
-- pelos arquivos 001/002. Só insere municípios/empresas novos.
------------------------------------------------------------

------------------------------------------------------------
-- MUNICÍPIOS (capitais dos estados ainda não cobertos)
-- Códigos IBGE reais.
------------------------------------------------------------

INSERT INTO dim_municipios
(
    codigo,
    nome,
    uf,
    capital,
    data_inclusao
)
SELECT v.codigo, v.nome, v.uf, v.capital, v.data_inclusao
FROM
(
    VALUES
    ('1200401', 'Rio Branco', 'AC', true, DATE '2016-04-11'),
    ('2704302', 'Maceió', 'AL', true, DATE '2016-09-23'),
    ('1600303', 'Macapá', 'AP', true, DATE '2017-01-30'),
    ('3205309', 'Vitória', 'ES', true, DATE '2017-06-12'),
    ('2507507', 'João Pessoa', 'PB', true, DATE '2018-02-19'),
    ('2211001', 'Teresina', 'PI', true, DATE '2018-07-26'),
    ('2408102', 'Natal', 'RN', true, DATE '2019-01-15'),
    ('1100205', 'Porto Velho', 'RO', true, DATE '2019-08-08'),
    ('1400100', 'Boa Vista', 'RR', true, DATE '2020-03-04'),
    ('1721000', 'Palmas', 'TO', true, DATE '2020-10-27')
) AS v(codigo, nome, uf, capital, data_inclusao)
WHERE NOT EXISTS
(
    SELECT 1 FROM dim_municipios m
    WHERE m.codigo = v.codigo
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
-- ===== Empresas com BAIXA em 202605 (alimentam "Baixas (último mês)") =====
-- Entram em competências anteriores e encerram exatamente na última competência.
('30000001000101', 'Madeireira Acre Verde LTDA', 'Acre Verde', 2, '0710301', '1200401', '202505', '202605', '202505', NULL, false, '2025-05-12'),
('30000002000102', 'Padaria Maceió Doce MEI', 'Maceió Doce', 1, '1091102', '2704302', '202506', '202605', '202506', NULL, true, '2025-06-15'),
('30000003000103', 'Esquadrias Amapá Norte LTDA', 'Amapá Norte', 2, '2512800', '1600303', '202507', '202605', '202507', NULL, false, '2025-07-09'),
('30000004000104', 'Resíduos Vitória Limpa LTDA', 'Vitória Limpa', 2, '3811400', '3205309', '202508', '202605', NULL, NULL, false, '2025-08-20'),
('30000005000105', 'Banco Paraíba Crédito LTDA', 'PB Crédito', 3, '6422100', '2507507', '202509', '202605', NULL, NULL, false, '2025-09-14'),
('30000006000106', 'Contábil Rondônia MEI', 'Contábil RO', 1, '6920601', '1100205', '202512', '202605', '202512', NULL, true, '2025-12-10'),
('30000007000107', 'Mineração Rio Branco LTDA', 'RB Mineração', 2, '0710301', '1200401', '202604', '202605', NULL, NULL, false, '2026-04-25'),
('30000008000108', 'Resíduos Teresina Eco LTDA', 'Teresina Eco', 2, '3811400', '2211001', '202505', '202605', NULL, NULL, false, '2025-05-30'),

-- ===== Empresas ativas (sem baixa) em novos estados =====
('30000009000109', 'Imóveis Teresina Center MEI', 'Teresina Center', 1, '6810201', '2211001', '202510', NULL, '202510', NULL, true, '2025-10-06'),
('30000010000110', 'Engenharia Potiguar LTDA', 'Potiguar Eng', 2, '7112000', '2408102', '202511', NULL, '202511', NULL, false, '2025-11-18'),
('30000011000111', 'Office Boa Vista LTDA', 'Office RR', 2, '8211300', '1400100', '202601', NULL, '202601', NULL, false, '2026-01-21'),
('30000012000112', 'Teatro Palmas Cena MEI', 'Palmas Cena', 1, '9001901', '1721000', '202602', NULL, '202602', NULL, true, '2026-02-13'),
('30000013000113', 'Auto Peças Vitória LTDA', 'Auto Vitória', 2, '4530703', '3205309', '202603', NULL, '202603', NULL, false, '2026-03-08'),
('30000014000114', 'Hotel Maceió Mar LTDA', 'Maceió Mar', 2, '5510801', '2704302', '202604', NULL, '202604', NULL, false, '2026-04-17'),

-- ===== Aberturas no último mês (202605) em novos estados =====
('30000015000115', 'Padaria Natal Pão MEI', 'Natal Pão', 1, '1091102', '2408102', '202605', NULL, '202605', NULL, true, '2026-05-09'),
('30000016000116', 'Imóveis João Pessoa MEI', 'JP Imóveis', 1, '6810201', '2507507', '202605', NULL, '202605', NULL, true, '2026-05-19');

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
