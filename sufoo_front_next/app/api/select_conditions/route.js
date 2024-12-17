import { createConnection } from '@/lib/db';

export async function GET(req) {
    try {
        const connection = await createConnection();

        const [healthConditions] = await connection.execute(
            `SELECT health_name, health_name_en FROM health_conditions`
        );

        const [drugs] = await connection.execute(
            `SELECT drug_name, drug_name_en FROM drugs`
        );

        const [supplements] = await connection.execute(
            `SELECT supplement_name, supplement_name_en FROM supplements`
        );

        const [specialConditions] = await connection.execute(
            `SELECT special_name, special_name_en FROM special_conditions`
        );

        await connection.end();

        return new Response(JSON.stringify({ 
            healthConditions, 
            drugs, 
            supplements, 
            specialConditions 
        }), { status: 200 });

    } catch (error) {
        return new Response(JSON.stringify({ message: `데이터 검색 실패: ${error.message}` }), { status: 500 });
    }
}