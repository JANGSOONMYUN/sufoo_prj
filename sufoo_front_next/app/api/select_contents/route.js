import { createConnection } from '@/lib/db';

export async function GET(req) {
    const url = req.nextUrl.searchParams.get('url');
    if (!url) {
        return new Response(JSON.stringify({ message: 'URL이 제공되지 않았습니다.', req }), { status: 400 });
      }
    try {
        const connection = await createConnection();

        // 테이블들 간 INNER JOIN을 통해 특정 URL에 대한 페이지 정보 획득
        const [rows] = await connection.execute(
            `SELECT pi.page_id, pi.search_words, pi.date, pi.advertise_info, c.data AS content_data, 
                    u.user_id, u.session_id, u.gender, u.weight, u.height
             FROM page_info pi
             INNER JOIN contents c ON pi.contents_id = c.contents_id
             INNER JOIN user u ON pi.user_id = u.user_id
             WHERE pi.url = ?`,
            [url]
        );

        await connection.end(); // Ensure the connection is closed

        if (rows.length === 0) {
            return new Response(JSON.stringify({ message: '페이지 정보를 찾을 수 없습니다.', url, rows, req }), { status: 404 });
        }

        return new Response(JSON.stringify({ pageInfo: rows[0] }), { status: 200 });

    } catch (error) {
        return new Response(JSON.stringify({ message: `데이터 검색 실패: ${error.message}` }), { status: 500 });
    }
}