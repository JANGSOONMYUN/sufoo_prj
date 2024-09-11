import React, { useState } from 'react';
import axios from 'axios';
import Home from './Home';
import SearchResults from './SearchResults';
import FetchData from './components/FetchData';

function App() {
  const [term, setTerm] = useState('');
  const [results, setResults] = useState(null);
  const handleSearch = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/search', {
        params: { term }
      });
      setResults(response.data);
    } catch (error) {
      console.error('Error fetching search results:', error);
    }
  };
  console.log('1111');
  return (
    <div>
      <h1>React and Flask Integration</h1>

      {/* FetchData 컴포넌트를 상단에 배치하여 데이터를 로드 */}
      <FetchData />

      {/* 검색 기능을 위한 Home 컴포넌트 */}
      <Home term={term} setTerm={setTerm} onSearch={handleSearch} />

      {/* 검색 결과를 렌더링하는 SearchResults 컴포넌트 */}
      {results && <SearchResults results={results} />}
    </div>
  );
}

export default App;
