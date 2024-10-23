// lib/db.js

import mysql from 'mysql2/promise';

// MariaDB 연결 설정
// const dbConfig = {
//     host: 'localhost',
//     port: '13307',
//     user: 'sufoo',
//     password: 'jin1206123',
//     database: 'sufoo'
// };

const dbConfig = {
    host: 'jsm0803.iptime.org',
    port: '13306',
    user: 'sufoo',
    password: 'jsm0803123',
    database: 'sufoo'
};

// DB 연결 생성 함수
export async function createConnection() {
    try {
        const connection = await mysql.createConnection(dbConfig);
        return connection;
    } catch (error) {
        throw new Error(`DB 연결 오류: ${error.message}`);
    }
}
