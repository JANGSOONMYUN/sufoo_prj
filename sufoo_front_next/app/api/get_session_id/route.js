import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function GET() {
    const cookieStore = cookies();
    let sessionId = cookieStore.get('sessionId')?.value;

    if (!sessionId) {
        // 세션 ID가 없으면 새로 생성
        sessionId = 'session_' + Date.now().toString();
        
        // 쿠키 설정 (7일 유효)
        const response = NextResponse.json({ sessionId });
        response.cookies.set('sessionId', sessionId, { 
            httpOnly: true, 
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'strict',
            maxAge: 60 * 60 * 24 * 7 // 7일
        });
        
        return response;
    }

    // 기존 세션 ID 반환
    return NextResponse.json({ sessionId });
}
