// app/api/select_user_data/route.js

import { createConnection } from '@/lib/db';

// 사용자 정보 조회 핸들러 (POST 요청)
export async function POST(req) {
    try {
        const body = await req.json(); // POST 요청에서 body 데이터 파싱
        const { userId } = body; // body에서 userId 추출
        console.log('body:', body); // body 데이터 확인
        if (!userId) {
            return new Response(JSON.stringify({ message: 'userId가 제공되지 않았습니다.' }), { status: 400 });
        }
        console.log('userId:', userId); // userId 값 확인

        const connection = await createConnection();

        // SQL 쿼리로 여러 테이블에서 데이터를 조회합니다.
        const query = `
            SELECT 
                u.user_id, u.session_id, u.gender, u.weight, u.height,
                pi.date, pi.url, pi.advertise_info,
                c.data AS content_data
            FROM 
                user u
            LEFT JOIN 
                page_info pi ON u.user_id = pi.user_id
            LEFT JOIN 
                contents c ON pi.contents_id = c.contents_id
            WHERE 
                u.user_id = ?;
        `;

        const [rows] = await connection.execute(query, [userId]);
        console.log('조회된 데이터:', rows); // 조회된 데이터 확인

        if (rows.length > 0) {
            return new Response(JSON.stringify(rows), { status: 200 });
        } else {
            return new Response(JSON.stringify({ message: '사용자를 찾을 수 없습니다.' ,
                                                body: body,
                                                userId: userId}), 
                                                { status: 404 });
        }
    } catch (error) {
        return new Response(JSON.stringify({ message: `데이터 조회 실패: ${error.message}` }), { status: 500 });
    }
}
