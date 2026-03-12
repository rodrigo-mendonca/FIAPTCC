#!/usr/bin/env python3
"""
Document Optimizer - Otimização de documentos salvos no ChromaDB
Reduz redundâncias e tokens mantendo relevância nas buscas
"""

from typing import List, Dict, Any, Optional
import re


class TextOptimizer:
    """Otimiza textos removendo redundâncias e truncando informações desnecessárias"""
    
    # Configurações padrão
    MAX_DESCRIPTION_LENGTH = 150  # Máximo de caracteres para descrições
    MAX_FIELD_LENGTH = 100  # Máximo para campos simples
    MIN_TEXT_LENGTH = 30  # Mínimo para manter um documento
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 150, keep_word_boundary: bool = True) -> str:
        """
        Trunca texto mantendo limites de palavra
        
        Args:
            text: Texto a truncar
            max_length: Comprimento máximo
            keep_word_boundary: Se True, não corta no meio de palavras
            
        Returns:
            Texto truncado
        """
        if not text or len(text) <= max_length:
            return text
        
        if not keep_word_boundary:
            return text[:max_length]
        
        # Trunca e encontra o último espaço
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.7:  # Se houver espaço razoável
            return truncated[:last_space].rstrip('.,:;')
        
        return truncated.rstrip('.,:;')
    
    @staticmethod
    def remove_redundancy(text: str, metadata: Dict) -> str:
        """
        Remove informações redundantes que já estão em metadados
        
        Args:
            text: Texto original
            metadata: Dicionário de metadados
            
        Returns:
            Texto sem redundâncias
        """
        # Remove repetições de valores que estão em metadados
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) < 50:
                # Remove exatas repetições do valor nos metadados
                text = re.sub(rf'\b{re.escape(value)}\b', '', text, flags=re.IGNORECASE)
        
        # Remove "Tipo: X" se 'type' está em metadados
        if 'type' in metadata:
            text = re.sub(r'Tipo:\s*\w+\.?\s*', '', text, flags=re.IGNORECASE)
        
        # Remove múltiplos pontos e espaços
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def optimize_text(text: str, metadata: Dict, max_length: int = 200) -> str:
        """
        Aplica múltiplas otimizações ao texto
        
        Args:
            text: Texto original
            metadata: Metadados do documento
            max_length: Comprimento máximo final
            
        Returns:
            Texto otimizado
        """
        if not text:
            return ""
        
        # Remove redundâncias
        text = TextOptimizer.remove_redundancy(text, metadata)
        
        # Trunca se necessário
        if len(text) > max_length:
            text = TextOptimizer.truncate_text(text, max_length)
        
        return text.strip()
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estima número de tokens (aproximadamente 1 token = 4 caracteres)
        
        Args:
            text: Texto para estimar
            
        Returns:
            Número estimado de tokens
        """
        # Aproximação: 1 token ≈ 0.75 palavras ≈ 4 caracteres
        return max(1, len(text) // 4)


class DocumentOptimizer:
    """Otimiza documentos antes de salvar no ChromaDB"""
    
    @staticmethod
    def optimize_table_document(table_name: str, file_data: Dict, 
                                max_text_length: int = 200) -> Optional[Dict]:
        """
        Otimiza documento de tabela
        
        Args:
            table_name: Nome da tabela
            file_data: Dados da tabela
            max_text_length: Comprimento máximo do texto
            
        Returns:
            Documento otimizado ou None se não viável
        """
        # Texto principal: apenas nome e descrição curta
        description = file_data.get('descricao_curta', '').strip()
        text = f"Tabela {table_name}"
        
        if description:
            text += f": {TextOptimizer.truncate_text(description, 120)}"
        
        # Metadados com informações completas
        metadata = {
            'type': 'table',
            'table_name': table_name,
            'database': file_data.get('database', ''),
            'total_registros': str(file_data.get('total_registros', '0')),
            'ultima_atualizacao': file_data.get('ultima_atualizacao', ''),
            'source': 'database_structure'
        }
        
        return {
            'id': f"table_{table_name.lower().replace(' ', '_')}",
            'text': text,
            'metadata': metadata
        }
    
    @staticmethod
    def optimize_column_documents(table_name: str, colunas: List[Dict], 
                                  max_text_length: int = 150) -> List[Dict]:
        """
        Otimiza documentos de colunas - cria apenas para colunas importantes
        
        Args:
            table_name: Nome da tabela
            colunas: Lista de colunas
            max_text_length: Comprimento máximo do texto
            
        Returns:
            Lista de documentos de colunas otimizados
        """
        documents = []
        
        if not isinstance(colunas, list):
            return documents
        
        for col_idx, coluna in enumerate(colunas):
            if not isinstance(coluna, dict):
                continue
            
            col_name = coluna.get('nome', f'col_{col_idx}')
            col_type = coluna.get('tipo', '')
            col_desc = coluna.get('descricao', '').strip()
            
            # Texto principal: nome, tipo e descrição curta
            text = f"{col_name} ({col_type})"
            
            if col_desc:
                text += f": {TextOptimizer.truncate_text(col_desc, 80)}"
            
            # Exemplo apenas em metadados se muito longo
            example = coluna.get('exemplo_significativo', '')
            
            metadata = {
                'type': 'column',
                'table_name': table_name,
                'column_name': col_name,
                'column_type': col_type,
                'example': example[:100] if example else '',  # Trunca exemplo
                'source': 'database_structure'
            }
            
            documents.append({
                'id': f"column_{table_name.lower()}_{col_name.lower()}",
                'text': text,
                'metadata': metadata
            })
        
        return documents
    
    @staticmethod
    def optimize_business_rule_document(nome: str, regra: Dict, 
                                       idx: int, max_text_length: int = 200) -> Optional[Dict]:
        """
        Otimiza documento de regra de negócio
        
        Args:
            nome: Nome da regra
            regra: Dados da regra
            idx: Índice da regra
            max_text_length: Comprimento máximo do texto
            
        Returns:
            Documento otimizado ou None se não viável
        """
        explicacao = regra.get('explicacao', '').strip()
        tipo_regra = regra.get('tipo', '').strip()
        prioridade = regra.get('prioridade', '').strip()
        
        # Texto compact: nome e explicação breve
        text = f"Regra: {nome}"
        
        if tipo_regra:
            text += f" ({tipo_regra})"
        
        if explicacao:
            # Trunca explicação mas mantém clareza
            short_explain = TextOptimizer.truncate_text(explicacao, 120)
            text += f" - {short_explain}"
        
        metadata = {
            'type': 'business_rule',
            'nome_regra': nome,
            'tipo_regra': tipo_regra,
            'prioridade': prioridade,
            'source': 'business_rules'
        }
        
        return {
            'id': f"regra_{nome.lower().replace(' ', '_')}_{idx}",
            'text': text,
            'metadata': metadata
        }
    
    @staticmethod
    def optimize_service_document(nome_servico: str, rotina: Dict, 
                                 idx: int, max_text_length: int = 200) -> Optional[Dict]:
        """
        Otimiza documento de rotina de sistema
        
        Args:
            nome_servico: Nome do serviço
            rotina: Dados da rotina
            idx: Índice
            max_text_length: Comprimento máximo do texto
            
        Returns:
            Documento otimizado ou None se não viável
        """
        descricao = rotina.get('descricao', '').strip()
        tipo_servico = rotina.get('tipo_servico', '').strip()
        frequencia = rotina.get('frequencia', '').strip()
        prioridade = rotina.get('prioridade', '').strip()
        
        # Texto compacto
        text = f"Serviço: {nome_servico}"
        
        if tipo_servico:
            text += f" ({tipo_servico})"
        
        if descricao:
            short_desc = TextOptimizer.truncate_text(descricao, 100)
            text += f" - {short_desc}"
        
        metadata = {
            'type': 'rotina_sistema',
            'nome_rotina': nome_servico,
            'tipo_servico': tipo_servico,
            'frequencia': frequencia,
            'prioridade': prioridade,
            'source': 'system_services'
        }
        
        return {
            'id': f"rotina_{nome_servico.lower().replace(' ', '_')}_{idx}",
            'text': text,
            'metadata': metadata
        }
    
    @staticmethod
    def optimize_user_routine_document(nome_rotina: str, rotina: Dict, 
                                      idx: int, max_text_length: int = 200) -> Optional[Dict]:
        """
        Otimiza documento de rotina de usuário
        
        Args:
            nome_rotina: Nome da rotina
            rotina: Dados da rotina
            idx: Índice
            max_text_length: Comprimento máximo do texto
            
        Returns:
            Documento otimizado ou None se não viável
        """
        descricao = rotina.get('descricao', '').strip()
        frequencia = rotina.get('frequencia', '').strip()
        tempo_estimado = rotina.get('tempo_estimado', '').strip()
        
        # Papéis e módulos em metadados, não no texto
        papeis_raw = rotina.get('papeis_necessarios', [])
        papeis = ",".join(papeis_raw) if isinstance(papeis_raw, list) else str(papeis_raw)
        
        modulos_raw = rotina.get('modulos_envolvidos', [])
        modulos = ",".join(modulos_raw) if isinstance(modulos_raw, list) else str(modulos_raw)
        
        # Texto compacto
        text = f"Rotina: {nome_rotina}"
        
        if descricao:
            short_desc = TextOptimizer.truncate_text(descricao, 100)
            text += f" - {short_desc}"
        
        metadata = {
            'type': 'rotina_usuario',
            'nome_rotina': nome_rotina,
            'frequencia': frequencia,
            'tempo_estimado': tempo_estimado,
            'papeis_necessarios': papeis[:200],  # Trunca se muito longo
            'modulos_envolvidos': modulos[:200],
            'source': 'rotinas_usuario'
        }
        
        return {
            'id': f"rotina_usuario_{nome_rotina.lower().replace(' ', '_')}_{idx}",
            'text': text,
            'metadata': metadata
        }


class TokenCounter:
    """Conta tokens e fornece relatórios de economia"""
    
    @staticmethod
    def count_tokens_in_documents(documents: List[Dict]) -> int:
        """
        Conta tokens totais em documentos
        
        Args:
            documents: Lista de documentos
            
        Returns:
            Número estimado total de tokens
        """
        total_tokens = 0
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Conta tokens no texto (1 token ≈ 4 chars)
            text_tokens = TextOptimizer.estimate_tokens(text)
            
            # Conta tokens nos metadados
            metadata_str = str(metadata)
            metadata_tokens = TextOptimizer.estimate_tokens(metadata_str)
            
            total_tokens += text_tokens + metadata_tokens
        
        return total_tokens
    
    @staticmethod
    def get_savings_report(original_documents: List[Dict], 
                          optimized_documents: List[Dict]) -> Dict[str, Any]:
        """
        Calcula economia de tokens entre documentos originais e otimizados
        
        Args:
            original_documents: Documentos originais
            optimized_documents: Documentos otimizados
            
        Returns:
            Relatório com estatísticas de economia
        """
        original_tokens = TokenCounter.count_tokens_in_documents(original_documents)
        optimized_tokens = TokenCounter.count_tokens_in_documents(optimized_documents)
        
        tokens_saved = original_tokens - optimized_tokens
        percent_saved = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
        
        return {
            'original_tokens': original_tokens,
            'optimized_tokens': optimized_tokens,
            'tokens_saved': tokens_saved,
            'percent_saved': round(percent_saved, 1),
            'document_count_original': len(original_documents),
            'document_count_optimized': len(optimized_documents),
            'avg_tokens_per_doc_original': round(original_tokens / len(original_documents)) if original_documents else 0,
            'avg_tokens_per_doc_optimized': round(optimized_tokens / len(optimized_documents)) if optimized_documents else 0
        }
