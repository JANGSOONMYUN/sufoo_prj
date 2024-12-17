// app/api/db/route.js

import { createConnection } from '@/lib/db';

export async function POST(req) {
    const { userData, healthIds, drugIds, supplementIds, specialIds } = await req.json();

    try {
        const connection = await createConnection();
        await connection.beginTransaction();
        
        console.log('22222222222222222222:' ); // 전송 데이터 로그
        // console.log('userData.gender,',userData ); // 전송 데이터 로그
        const gender = userData.gender || '';
        const weight = userData.weight || 0;  // 기본값을 0으로 설정
        const height = userData.height || 0;  // 기본값을 0으로 설정        
        

        // user 테이블에 데이터 삽입
        const [result] = await connection.execute(
            `INSERT INTO user (session_id, gender, weight, height) 
             VALUES (?, ?, ?, ?)`,
            [userData.session_id, '남', '32', '175']
        );
        const userId = result.insertId;

        console.log('333333333333333333:' ); // 전송 데이터 로그

        // user_health, user_drug, user_supplement, user_special_cond 테이블에 데이터 삽입
        const insertHealthQuery = `INSERT INTO user_health (user_id, health_id) VALUES (?, ?)`;
        const insertDrugQuery = `INSERT INTO user_drug (user_id, drug_id) VALUES (?, ?)`;
        const insertSupplementQuery = `INSERT INTO user_supplement (user_id, supplement_id) VALUES (?, ?)`;
        const insertSpecialQuery = `INSERT INTO user_special_cond (user_id, special_id) VALUES (?, ?)`;

        console.log('44444444444444444:' ); // 전송 데이터 로그

        for (const healthId of healthIds) {
            await connection.execute(insertHealthQuery, [userId, healthId]);
        }
        for (const drugId of drugIds) {
            await connection.execute(insertDrugQuery, [userId, drugId]);
        }
        for (const supplementId of supplementIds) {
            await connection.execute(insertSupplementQuery, [userId, supplementId]);
        }
        for (const specialId of specialIds) {
            await connection.execute(insertSpecialQuery, [userId, specialId]);
        }

        await connection.commit();
        return new Response(JSON.stringify({ message: '사용자 데이터 삽입 성공', userId }), { status: 200 });

    } catch (error) {
        return new Response(JSON.stringify({ message: `데이터 삽입 실패: ${error.message}` }), { status: 500 });
    }
}
