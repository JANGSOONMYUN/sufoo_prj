import { createConnection } from '@/lib/db';

export const revalidate = 0;

export async function GET(req) {
    try {
        // 추천 검색어를 DB에서 가져오는 요청
        const connection = await createConnection();

        await connection.execute('SET SESSION query_cache_type = OFF');        

        const [recommendData] = await connection.execute(
            `SELECT SQL_NO_CACHE DISTINCT search_words 
             FROM page_info 
             WHERE LENGTH(search_words) < 40 
             AND search_words NOT REGEXP '^[0-9]+$' 
             AND search_words NOT LIKE '%검색%'
             AND LENGTH(search_words) > 6 
             ORDER BY RAND(UNIX_TIMESTAMP() * CONNECTION_ID())
             LIMIT 3;`
        );

        await connection.end();
        
        return new Response(JSON.stringify({
            recommendData
        }), {
             status: 200,
             headrs: {
                'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        });
    } catch (error) {
        return new Response(JSON.stringify({ message: `데이터 검색 실패: ${error.message}` }), { status: 500 });
    }
}