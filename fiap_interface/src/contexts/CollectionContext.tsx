import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

const API_URL = process.env.REACT_APP_API_URL;

interface CollectionContextType {
  selectedCollection: string;
  setSelectedCollection: (collection: string) => void;
  availableCollections: Array<{
    name: string;
    count: number;
    id: string;
  }>;
  setAvailableCollections: (collections: Array<{
    name: string;
    count: number;
    id: string;
  }>) => void;
  refreshCollections: (collectionNameToUse?: string) => Promise<void>;
  createCollection: (collectionName: string) => Promise<void>;
  deleteCollection: (collectionName: string) => Promise<void>;
}

const CollectionContext = createContext<CollectionContextType | undefined>(undefined);

interface CollectionProviderProps {
  children: ReactNode;
}

export const CollectionProvider: React.FC<CollectionProviderProps> = ({ children }) => {
  const [selectedCollection, setSelectedCollectionState] = useState<string>('');
  const [availableCollections, setAvailableCollections] = useState<Array<{
    name: string;
    count: number;
    id: string;
  }>>([]);

  // Carrega coleção do localStorage na inicialização
  useEffect(() => {
    const savedCollection = localStorage.getItem('selectedCollection');
    if (savedCollection) {
      setSelectedCollectionState(savedCollection);
    }
  }, []);

  // Função para definir coleção (com localStorage e reload)
  const setSelectedCollection = (collection: string) => {
    setSelectedCollectionState(collection);
    localStorage.setItem('selectedCollection', collection);
    // Recarrega a página para limpar todos os campos e aplicar a nova coleção
    window.location.reload();
  };

  // Função para atualizar a lista de coleções disponíveis
  const refreshCollections = async (collectionNameToUse?: string) => {
    try {
      // Sempre passa collection_name vazio para obter TODAS as coleções
      // Isso garante que a lista sempre fica atualizada
      const response = await fetch(`${API_URL}/vectordb/stats?collection_name=`);
      if (response.ok) {
        const data = await response.json();
        if (data.collections) {
          console.log('✓ Coleções atualizadas:', data.collections.length);
          setAvailableCollections(data.collections);
        }
      } else {
        console.error('Erro ao atualizar coleções:', response.status);
      }
    } catch (error) {
      console.error('Erro ao atualizar coleções:', error);
    }
  };

  // Carrega coleções na inicialização
  useEffect(() => {
    refreshCollections();
  }, [selectedCollection]);

  // Carrega coleções periodicamente para manter sincronizado
  useEffect(() => {
    const interval = setInterval(() => {
      refreshCollections();
    }, 5000); // Atualiza a cada 5 segundos
    
    return () => clearInterval(interval);
  }, [selectedCollection]);

  // Função para criar uma nova coleção
  const createCollection = async (collectionName: string) => {
    try {
      const response = await fetch(`${API_URL}/vectordb/create-collection`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: collectionName }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erro ao criar coleção');
      }

      // Atualiza a coleção selecionada para a nova
      setSelectedCollection(collectionName);
      // Recarrega a página para aplicar a mudança
      await new Promise(resolve => setTimeout(resolve, 500));
      window.location.reload();
    } catch (error) {
      throw error;
    }
  };

  // Função para deletar uma coleção
  const deleteCollection = async (collectionName: string) => {
    try {
      const response = await fetch(`${API_URL}/vectordb/collection/${encodeURIComponent(collectionName)}/delete`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erro ao deletar coleção');
      }

      // Se a coleção deletada era a atual, redefine para vazia
      if (selectedCollection === collectionName) {
        setSelectedCollectionState('');
        try {
          localStorage.setItem('selectedCollection', '');
        } catch (e) {
          // ignore
        }
      }
      
      // Atualiza a lista de coleções após deletar (passa qualquer nome só pra fazer a chamada)
      await refreshCollections('');
    } catch (error) {
      throw error;
    }
  };

  const value: CollectionContextType = {
    selectedCollection,
    setSelectedCollection,
    availableCollections,
    setAvailableCollections,
    refreshCollections,
    createCollection,
    deleteCollection,
  };

  return (
    <CollectionContext.Provider value={value}>
      {children}
    </CollectionContext.Provider>
  );
};

export const useCollection = () => {
  const context = useContext(CollectionContext);
  if (context === undefined) {
    throw new Error('useCollection must be used within a CollectionProvider');
  }
  return context;
};