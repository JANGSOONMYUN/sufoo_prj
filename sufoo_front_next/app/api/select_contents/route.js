import { createConnection } from '@/lib/db';

export async function GET(req) {
    const url = new URL(req.url).searchParams.get('url');
    if (!url) {
        return new Response(JSON.stringify({ message: 'URL이 제공되지 않았습니다.', req }), { status: 400 });
    }

    const startTime = Date.now(); // Start timing

    try {
        const connection = await createConnection();

        const [rows] = await connection.execute(
            `SELECT pi.page_id, pi.search_words, pi.date, pi.advertise_info, c.data AS content_data, 
                    u.user_id, u.session_id, u.gender, u.weight, u.height
             FROM page_info pi
             INNER JOIN contents c ON pi.contents_id = c.contents_id
             INNER JOIN user u ON pi.user_id = u.user_id
             WHERE pi.url = ?`,
            [url]
        );

        await connection.end();

        const endTime = Date.now(); // End timing
        const elapsedTime = endTime - startTime; // Calculate elapsed time

        if (rows.length === 0) {
            return new Response(JSON.stringify({ message: '페이지 정보를 찾을 수 없습니다.', url, rows, req, time: `${elapsedTime}ms` }), { status: 404 });
        }

        return new Response(JSON.stringify({ pageInfo: rows[0], time: `${elapsedTime}ms` }), { status: 200 });

    } catch (error) {
        const endTime = Date.now(); // End timing
        const elapsedTime = endTime - startTime; // Calculate elapsed time

        return new Response(JSON.stringify({ message: `데이터 검색 실패: ${error.message}`, time: `${elapsedTime}ms` }), { status: 500 });
    }
}