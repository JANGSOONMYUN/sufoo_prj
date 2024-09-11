// src/components/FetchData.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const FetchData = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    //console.log('진입');

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('http://localhost:5000/api/endpoint');
                setData(response.data);  // API에서 받은 데이터를 상태로 저장
                setLoading(false);  // 데이터 로딩이 끝났음을 표시
            } catch (err) {
                setError(err);
                setLoading(false);
            }
        };  

        //console.log('진입22');

        fetchData();
    }, []);  // 컴포넌트가 처음 렌더링될 때만 실행

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error.message}</div>;

    return (
        <div>
            <h1>Fetched Data:</h1>
            <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
    );
};

export default FetchData;
