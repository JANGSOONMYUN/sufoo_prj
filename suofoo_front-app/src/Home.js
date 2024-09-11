import React from 'react';

function Home({ term, setTerm, onSearch }) {
  return (
    <div>
      <input
        type="text"
        value={term}
        onChange={(e) => setTerm(e.target.value)}
      />
      <button onClick={onSearch}>Search</button>
    </div>
  );
}

export default Home;
