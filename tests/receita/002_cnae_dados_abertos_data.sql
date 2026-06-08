BEGIN;

------------------------------------------------------------
-- DADOS FICTÍCIOS ADICIONAIS (NÃO LIMPA AS TABELAS)
-- As dimensões compartilhadas só são inseridas se ainda
-- não existirem, reaproveitando os IDs já cadastrados.
------------------------------------------------------------

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
SELECT v.competencia, v.atualizado, v.data_inclusao
FROM
(
    VALUES
    ('202505', true, DATE '2025-06-10'),
    ('202506', true, DATE '2025-07-13'),
    ('202507', true, DATE '2025-08-11'),
    ('202508', true, DATE '2025-09-19'),
    ('202509', true, DATE '2025-10-07'),
    ('202510', true, DATE '2025-11-15'),
    ('202511', true, DATE '2025-12-06'),
    ('202512', true, DATE '2026-01-12'),
    ('202601', true, DATE '2026-02-24'),
    ('202602', true, DATE '2026-03-11'),
    ('202603', true, DATE '2026-04-14'),
    ('202604', true, DATE '2026-05-19'),
    ('202605', true, DATE '2026-06-04')
) AS v(competencia, atualizado, data_inclusao)
WHERE NOT EXISTS
(
    SELECT 1 FROM dim_competencias d
    WHERE d.competencia = v.competencia
);

------------------------------------------------------------
-- SEGMENTOS CNAE - SEÇÃO OFICIAL
------------------------------------------------------------

INSERT INTO dim_cnaes_segmento
(
    secao,
    denominacao,
    data_inclusao
)
SELECT v.secao, v.denominacao, v.data_inclusao
FROM
(
    VALUES
    ('B', 'Indústrias extrativas', DATE '2018-04-09'),
    ('C', 'Indústrias de transformação', DATE '2017-08-14'),
    ('E', 'Água, esgoto, atividades de gestão de resíduos e descontaminação', DATE '2019-06-27'),
    ('K', 'Atividades financeiras, de seguros e serviços relacionados', DATE '2020-02-19'),
    ('L', 'Atividades imobiliárias', DATE '2018-12-05'),
    ('M', 'Atividades profissionais, científicas e técnicas', DATE '2021-01-22'),
    ('N', 'Atividades administrativas e serviços complementares', DATE '2019-09-30'),
    ('R', 'Artes, cultura, esporte e recreação', DATE '2020-07-16'),
    ('G', 'Comércio; reparação de veículos automotores e motocicletas', DATE '2017-11-30'),
    ('I', 'Alojamento e alimentação', DATE '2021-02-17')
) AS v(secao, denominacao, data_inclusao)
WHERE NOT EXISTS
(
    SELECT 1 FROM dim_cnaes_segmento s
    WHERE s.secao = v.secao
);

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
SELECT v.codigo, v.nome, s.id, v.data_inclusao
FROM
(
    VALUES
    ('0710301', 'Extração de minério de ferro', 'B', DATE '2019-03-18'),
    ('1091102', 'Fabricação de produtos de padaria e confeitaria com predominância de produção própria', 'C', DATE '2018-10-22'),
    ('2512800', 'Fabricação de esquadrias de metal', 'C', DATE '2020-05-14'),
    ('3811400', 'Coleta de resíduos não perigosos', 'E', DATE '2019-11-03'),
    ('6422100', 'Bancos múltiplos, com carteira comercial', 'K', DATE '2021-06-08'),
    ('6810201', 'Compra e venda de imóveis próprios', 'L', DATE '2019-02-26'),
    ('7112000', 'Serviços de engenharia', 'M', DATE '2020-09-17'),
    ('6920601', 'Atividades de contabilidade', 'M', DATE '2018-08-30'),
    ('8211300', 'Serviços combinados de escritório e apoio administrativo', 'N', DATE '2021-04-12'),
    ('9001901', 'Produção teatral', 'R', DATE '2020-11-25'),
    ('4530703', 'Comércio a varejo de peças e acessórios novos para veículos automotores', 'G', DATE '2018-01-19'),
    ('5510801', 'Hotéis', 'I', DATE '2021-07-29')
) AS v(codigo, nome, secao, data_inclusao)
JOIN dim_cnaes_segmento s
    ON s.secao = v.secao
WHERE NOT EXISTS
(
    SELECT 1 FROM dim_cnaes dc
    WHERE dc.codigo = v.codigo
);

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
SELECT v.codigo, v.nome, v.uf, v.capital, v.data_inclusao
FROM
(
    VALUES
    ('2611606', 'Recife', 'PE', true, DATE '2015-06-15'),
    ('1302603', 'Manaus', 'AM', true, DATE '2015-08-09'),
    ('1501402', 'Belém', 'PA', true, DATE '2016-02-21'),
    ('4314902', 'Porto Alegre', 'RS', true, DATE '2016-07-04'),
    ('4205407', 'Florianópolis', 'SC', true, DATE '2017-03-26'),
    ('5300108', 'Brasília', 'DF', true, DATE '2015-12-01'),
    ('3543402', 'Ribeirão Preto', 'SP', false, DATE '2017-10-13'),
    ('4209102', 'Joinville', 'SC', false, DATE '2018-04-28'),
    ('4304606', 'Caxias do Sul', 'RS', false, DATE '2018-09-15'),
    ('2607901', 'Caruaru', 'PE', false, DATE '2019-05-07'),
    ('1303403', 'Parintins', 'AM', false, DATE '2019-11-22'),
    ('3518800', 'Guarulhos', 'SP', false, DATE '2020-03-30'),
    ('3552205', 'Sorocaba', 'SP', false, DATE '2020-10-18'),
    ('2111300', 'São Luís', 'MA', true, DATE '2021-05-09'),
    ('2800308', 'Aracaju', 'SE', true, DATE '2021-12-02')
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
-- Existentes em 05/2025
('20000001000101', 'Minérios Vale Sul LTDA', 'Vale Sul', 3, '0710301', '4314902', '202505', NULL, NULL, NULL, false, '2025-05-08'),
('20000002000102', 'Forno Dourado Indústria LTDA', 'Forno Dourado', 2, '1091102', '2611606', '202505', NULL, '202505', NULL, false, '2025-05-16'),
('20000003000103', 'Metalúrgica Esquadrias Norte LTDA', 'Esquadrias Norte', 2, '2512800', '1302603', '202505', NULL, '202505', NULL, false, '2025-05-22'),
('20000004000104', 'EcoColeta Ambiental LTDA', 'EcoColeta', 2, '3811400', '1501402', '202505', NULL, NULL, NULL, false, '2025-05-04'),
('20000005000105', 'Imóveis Praça Central LTDA', 'Praça Central', 1, '6810201', '4205407', '202505', '202603', '202505', '202602', true, '2025-05-27'),
('20000006000106', 'Engenharia Estrutural Plena LTDA', 'EngPlena', 3, '7112000', '5300108', '202505', NULL, NULL, NULL, false, '2025-05-11'),
('20000007000107', 'Contábil Precisão LTDA', 'Contábil Precisão', 1, '6920601', '3543402', '202505', NULL, '202505', NULL, true, '2025-05-19'),
('20000008000108', 'Auto Peças Joinville LTDA', 'Auto Peças JLV', 2, '4530703', '4209102', '202505', NULL, '202505', NULL, false, '2025-05-06'),

-- Entradas em 2025
('20000009000109', 'Hotel Serra Azul LTDA', 'Serra Azul', 2, '5510801', '4304606', '202506', NULL, '202506', NULL, false, '2025-06-09'),
('20000010000110', 'Teatro Cultura Viva LTDA', 'Cultura Viva', 1, '9001901', '2607901', '202506', NULL, '202506', NULL, true, '2025-06-21'),
('20000011000111', 'Apoio Office Serviços LTDA', 'Apoio Office', 2, '8211300', '3518800', '202507', NULL, '202507', NULL, false, '2025-07-10'),
('20000012000112', 'Padaria Trigo de Ouro LTDA', 'Trigo de Ouro', 1, '1091102', '3552205', '202507', NULL, '202507', NULL, true, '2025-07-24'),
('20000013000113', 'Recicla Mais Resíduos LTDA', 'Recicla Mais', 2, '3811400', '2611606', '202508', NULL, NULL, NULL, false, '2025-08-13'),
('20000014000114', 'Banco Digital Aurora LTDA', 'Aurora Bank', 3, '6422100', '5300108', '202508', NULL, NULL, NULL, false, '2025-08-29'),
('20000015000115', 'Pousada Beira Rio LTDA', 'Beira Rio', 1, '5510801', '1303403', '202509', '202604', '202509', NULL, true, '2025-09-05'),
('20000016000116', 'Esquadrias Premium Sul LTDA', 'Esquadrias Premium', 2, '2512800', '4314902', '202509', NULL, '202509', NULL, false, '2025-09-17'),
('20000017000117', 'Engenharia Costa Verde LTDA', 'Costa Verde', 2, '7112000', '2111300', '202510', NULL, '202510', NULL, false, '2025-10-09'),
('20000018000118', 'Cena Aberta Produções LTDA', 'Cena Aberta', 1, '9001901', '2800308', '202510', NULL, '202510', NULL, true, '2025-10-23'),
('20000019000119', 'Contabilidade Horizonte LTDA', 'Conta Horizonte', 2, '6920601', '4205407', '202511', NULL, '202511', NULL, false, '2025-11-05'),
('20000020000120', 'Auto Center Guarulhos LTDA', 'Auto Center GRU', 2, '4530703', '3518800', '202511', NULL, NULL, NULL, false, '2025-11-19'),
('20000021000121', 'Imobiliária Lar Feliz LTDA', 'Lar Feliz', 1, '6810201', '3543402', '202512', NULL, '202512', NULL, true, '2025-12-08'),
('20000022000122', 'Mineração Norte Forte LTDA', 'Norte Forte', 2, '0710301', '1302603', '202512', NULL, NULL, NULL, false, '2025-12-22'),

-- Entradas em 2026
('20000023000123', 'Office Suporte Capital LTDA', 'Office Capital', 2, '8211300', '5300108', '202601', NULL, '202601', NULL, false, '2026-01-14'),
('20000024000124', 'Fintech Crédito Fácil LTDA', 'Crédito Fácil', 2, '6422100', '4314902', '202601', NULL, NULL, NULL, false, '2026-01-28'),
('20000025000125', 'Resíduos Limpos Caxias LTDA', 'Resíduos Limpos', 2, '3811400', '4304606', '202602', NULL, NULL, NULL, false, '2026-02-05'),
('20000026000126', 'Pão Quente Caruaru LTDA', 'Pão Quente', 1, '1091102', '2607901', '202602', NULL, '202602', NULL, true, '2026-02-18'),
('20000027000127', 'Cênica Eventos MEI', 'Cênica Eventos', 1, '9001901', '3552205', '202603', NULL, '202603', NULL, true, '2026-03-11'),
('20000028000128', 'Metal Forte Esquadrias LTDA', 'Metal Forte', 2, '2512800', '4209102', '202603', NULL, '202603', NULL, false, '2026-03-26'),
('20000029000129', 'Engenharia Vale do Sol LTDA', 'Vale do Sol', 2, '7112000', '4314902', '202604', NULL, '202604', NULL, false, '2026-04-09'),
('20000030000130', 'Hotel Marés Nordeste LTDA', 'Hotel Marés', 2, '5510801', '2800308', '202604', NULL, '202604', NULL, false, '2026-04-20'),
('20000031000131', 'Contábil Express MEI', 'Contábil Express', 1, '6920601', '3543402', '202604', NULL, '202604', NULL, true, '2026-04-29'),
('20000032000132', 'Imóveis Vista Mar LTDA', 'Vista Mar', 1, '6810201', '2111300', '202605', NULL, '202605', NULL, true, '2026-05-06'),
('20000033000133', 'Banco Pix Investimentos LTDA', 'Banco Pix', 2, '6422100', '4205407', '202605', NULL, NULL, NULL, false, '2026-05-15'),
('20000034000134', 'Peças e Acessórios Sorocaba LTDA', 'Peças Sorocaba', 2, '4530703', '3552205', '202605', NULL, '202605', NULL, false, '2026-05-22'),
('20000035000135', 'Minério Bom Futuro LTDA', 'Bom Futuro', 2, '0710301', '1303403', '202605', NULL, NULL, NULL, false, '2026-05-28'),
('20000036000136', 'Coworking Apoio Total MEI', 'Apoio Total', 1, '8211300', '5300108', '202605', NULL, '202605', NULL, true, '2026-05-31');

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
