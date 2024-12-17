import { createConnection } from '@/lib/db';

function generateRandomString(length) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

async function isUniqueUrl(connection, url) {
    const [rows] = await connection.execute(`SELECT EXISTS(SELECT 1 FROM page_info WHERE url = ?) AS exist`, [url]);
    return !rows[0].exist;
}

export async function POST(req) {
    const { userId, searchWords, advertiseInfo, data, url } = await req.json();

    try {
        const connection = await createConnection();
        await connection.beginTransaction();

        // contents 테이블에 데이터 삽입
        const [contentsResult] = await connection.execute(
            `INSERT INTO contents (data) VALUES (?)`,
            [data]
        );
        const contentsId = contentsResult.insertId;


        const date = new Date().toISOString().slice(0, 19).replace('T', ' ');

        // page_info 테이블에 데이터 삽입
        await connection.execute(
            `INSERT INTO page_info (contents_id, user_id, search_words, date, url, advertise_info) 
             VALUES (?, ?, ?, ?, ?, ?)`,
            [contentsId, userId, searchWords, date, url, advertiseInfo]
        );

        await connection.commit();
        return new Response(JSON.stringify({ message: '페이지 정보 삽입 성공', pageId: contentsId }), { status: 200 });

    } catch (error) {
        return new Response(JSON.stringify({ message: `데이터 삽입 실패: ${error.message}` }), { status: 500 });
    }
}