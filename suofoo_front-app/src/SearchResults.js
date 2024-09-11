import React from 'react';

function SearchResults({ results }) {
  return (
    <div>
      <h2>Search Results:</h2>
      <p>{results.message}</p>
    </div>
  );
}

export default SearchResults;
