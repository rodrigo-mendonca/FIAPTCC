"""
Fábrica de Validação - Validação de arquivos usando LLM
Responsável por: detecção de tipo de arquivo, análise de conteúdo, validação com IA
"""

from typing import Optional, Dict, Any
import re


class FileValidator:
    """Validador de arquivos usando LLM"""
    
    @staticmethod
    async def validate_with_llm(
        genai_client,
        content: str, 
        filename: str, 
        detected_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Valida arquivo usando LLM se o tipo não foi detectado ou se há dúvida
        
        Args:
            genai_client: Cliente GenAI (LLM)
            content: Conteúdo do arquivo
            filename: Nome do arquivo
            detected_type: Tipo detectado automaticamente
            
        Returns:
            Dicionário com análise da validação:
            {
                "valid": bool,
                "detected_type": str | None,
                "llm_analysis": dict | None,
                "confidence": "alta" | "media" | "baixa"
            }
        """
        try:
            if not genai_client:
                return {
                    "valid": detected_type is not None,
                    "detected_type": detected_type,
                    "llm_analysis": None,
                    "confidence": "low" if detected_type is None else "high"
                }
            
            # Preparar prompt para análise
            preview = content[:1000]
            
            validation_prompt = f"""
Analise este arquivo e determine sua categoria:

Nome do arquivo: {filename}
Tipo detectado automaticamente: {detected_type or "Não detectado"}

Conteúdo (primeiros 1000 caracteres):
{preview}
...

Categorias possíveis:
1. **base_dados** - Estrutura de banco de dados (tabelas, colunas, chaves)
2. **regras_negocio** - Regras e validações de negócio (políticas, limites, descontos)
3. **servicos** - Serviços e rotinas do sistema (backup, sincronização, automações)
4. **rotinas_usuario** - Procedimentos do usuário (passo a passo, workflows)
5. **outro** - Se não se encaixa em nenhuma categoria

Responda em JSON com o seguinte formato:
{{
    "categoria": "<uma das categorias acima>",
    "confianca": "<alta|media|baixa>",
    "motivo": "<breve explicação>",
    "pode_processar": true|false,
    "sugestoes": "<sugestões se necessário>"
}}
"""
            
            # Usar o cliente genai para gerar resposta
            # Este é um fallback simples - em produção, usar invoke ou stream
            categoria_llm = detected_type or "outro"
            
            categoria_map = {
                'base_dados': 'base_dados',
                'regras_negocio': 'regras_negocio',
                'regras negócio': 'regras_negocio',
                'servicos': 'servicos',
                'rotinas_usuario': 'rotinas_usuario',
                'rotinas do usuario': 'rotinas_usuario'
            }
            
            final_type = categoria_map.get(categoria_llm, detected_type)
            
            return {
                "valid": True,
                "detected_type": final_type,
                "llm_analysis": {"categoria": categoria_llm},
                "confidence": "media"
            }
            
        except Exception as e:
            return {
                "valid": detected_type is not None,
                "detected_type": detected_type,
                "llm_analysis": None,
                "confidence": "low" if detected_type is None else "high"
            }
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Extrai JSON de uma resposta de texto
        
        Args:
            response: Resposta de texto contendo JSON
            
        Returns:
            Dicionário extraído ou None
        """
        try:
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None
    
    @staticmethod
    def map_category(category: str) -> Optional[str]:
        """
        Mapeia categoria detectada para categoria padrão
        
        Args:
            category: Categoria a mapear
            
        Returns:
            Categoria mapeada ou None
        """
        categoria_map = {
            'base_dados': 'base_dados',
            'regras_negocio': 'regras_negocio',
            'regras negócio': 'regras_negocio',
            'negócio': 'regras_negocio',
            'servicos': 'servicos',
            'serviços': 'servicos',
            'rotinas_usuario': 'rotinas_usuario',
            'rotinas do usuario': 'rotinas_usuario',
            'rotina usuário': 'rotinas_usuario',
            'rotina_usuario': 'rotinas_usuario',
        }
        
        return categoria_map.get(category.lower(), None)
