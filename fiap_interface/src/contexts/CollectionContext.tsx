import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

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
  refreshCollections: () => Promise<void>;
}

const CollectionContext = createContext<CollectionContextType | undefined>(undefined);

interface CollectionProviderProps {
  children: ReactNode;
}

export const CollectionProvider: React.FC<CollectionProviderProps> = ({ children }) => {
  const [selectedCollection, setSelectedCollectionState] = useState<string>('sistema_comercial');
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
  const refreshCollections = async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/vectordb/stats?collection_name=${selectedCollection}`);
      if (response.ok) {
        const data = await response.json();
        if (data.collections) {
          setAvailableCollections(data.collections);
        }
      }
    } catch (error) {
      console.error('Erro ao atualizar coleções:', error);
    }
  };

  // Carrega coleções na inicialização
  useEffect(() => {
    refreshCollections();
  }, [selectedCollection]);

  const value: CollectionContextType = {
    selectedCollection,
    setSelectedCollection,
    availableCollections,
    setAvailableCollections,
    refreshCollections,
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