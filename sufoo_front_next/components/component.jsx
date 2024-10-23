"use client"; // 클라이언트 컴포넌트로 지정

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import { useEffect } from "react";
import { usePathname , useRouter } from "next/navigation"; // useRouter 임포트 추가

export default function Component() {
  const pathname = usePathname();
  const router = useRouter(); // useRouter 훅 초기화

  useEffect(() => {

    if (router.isReady){

    }
  }, [router]);
  const [selectedGender, setSelectedGender] = useState(null);
  const [selectedHealthConditions, setSelectedHealthConditions] = useState([]);
  const [selectedSupplements, setSelectedSupplements] = useState([]);
  const [selectedSpecialNotes, setSelectedSpecialNotes] = useState([]);
  const [weight, setWeight] = useState(""); // 체중 상태 추가
  const [height, setHeight] = useState(""); // 키 상태 추가
  const [age, setAge] = useState(""); // 연령 상태 추가
  const [searchTerm, setSearchTerm] = useState(""); // 검색어 상태 추가
  const [loading, setLoading] = useState(false); // API 요청 상태 표시

  const toggleSelection = (category, item) => {
    const setSelectedFunction = {
      health: setSelectedHealthConditions,
      supplements: setSelectedSupplements,
      specialNotes: setSelectedSpecialNotes,
    };

    setSelectedFunction[category]((prevSelected) =>
      prevSelected.includes(item)
        ? prevSelected.filter((i) => i !== item)
        : [...prevSelected, item]
    );
  };

   // API 요청 보내기
   const handleSubmit = async () => {
    setLoading(true);
    setMessage('');

    // 입력 데이터 준비
    const formData = {
      session_id: sessionId,
      gender: selectedGender,
      weight,
      height,
      age,
      searchTerm, // 검색어 추가
      healthConditions: selectedHealthConditions,
      supplements: selectedSupplements,
      specialNotes: selectedSpecialNotes,
    };

    const body = {
        userData,
        healthIds,
        drugIds,
        supplementIds,
        specialIds
    };

    try {

        // 우선 ui_loading 화면으로 이동
        router.push('/ui_loading');


        // API 요청
        const res = await fetch('/api/insert_user_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await res.json();
        if (res.ok) {
            setMessage(`사용자 정보가 성공적으로 저장되었습니다! 사용자 ID: ${data.userId}`);
            
            // LLM을 위한 JSON 데이터 생성
            const llmJson = createLlmJson(formData, healthIds, drugIds, supplementIds, specialIds);
            setLlmJsonData(JSON.stringify(llmJson, null, 2));

            // 랜덤 문자열 생성
            const randomPageId = generateRandomString(10);

            // // 디버깅을 위한 로그 추가
            // console.log('Navigating to:', `/content_page/${randomPageId}`);
            // console.log('Query:', {
            //     userId: String(data.userId),
            //     sessionId: String(sessionId),
            //     llmJsonData: JSON.stringify(llmJson)
            // });

            // 쿼리 문자열 생성
            const query = new URLSearchParams({
                userId: String(data.userId),
                sessionId: String(sessionId),
                relative_url: String(randomPageId),
                llmJsonData: JSON.stringify(llmJson)
            }).toString();
            // 전체 URL 생성
            const url = `/content_page/${randomPageId}?${query}`;
            //const url = `/ui_loading`;
            // 페이지 이동
            router.push(url);

            // 이 방법은 13버전에서 지원되지 않는다
            // // 전체 URL 생성
            // const url = `/content_page/${randomPageId}`;
            // router.push(url, { query });

        } else {
            setMessage(`에러 발생: ${data.message}`);
            alert('에러가 발생했습니다. 다시 검색 부탁드립니다.');
            router.back(); // 이전 페이지로 돌아감
        }
      } catch (error) {
          setMessage(`API 호출 실패: ${error.message}`);
          alert('에러가 발생했습니다. 다시 검색 부탁드립니다.');
          router.back(); // 이전 페이지로 돌아감
      } finally {
          setLoading(false);
      }
    };

  

  return (
    <div className="flex flex-col items-center w-full min-h-screen p-4">
      <header className="flex items-center w-full px-4 py-2">
        <div className="flex-1">
          <h1 className="text-lg font-bold">SUFOO Logo</h1>
        </div>
      </header>
      <main className="w-full max-w-2xl">
        <section className="text-center">
          <h2 className="text-3xl font-bold">당신의 건강을 위한 영양 검색!</h2>
          <p className="text-muted-foreground">설명영역설명12315554</p>
        </section>
        <section className="mt-4">
          <div className="relative">
            <Input
              type="search"
              placeholder="당신에게 좋은 음식은?"
              className="w-full pl-8 pr-12 py-2"
              value={searchTerm} // 검색어 상태 연결
              onChange={(e) => setSearchTerm(e.target.value)} // 검색어 상태 업데이트
            />
            <ArrowRightIcon className="absolute right-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          </div>
          <div className="flex justify-between mt-2">
            <Button variant="outline" onClick={handleSubmit} disabled={loading}>
              {loading ? '저장 중...' : '데이터 저장'}
            </Button>
            <Button variant="outline" className="flex items-center space-x-1">
              상세 검색
              <ChevronDownIcon className="h-4 w-4" />
            </Button>
          </div>
        </section>
        <section className="mt-4">
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Label htmlFor="gender">성별</Label>
              <Button
                variant={selectedGender === "남" ? "default" : "outline"}
                onClick={() => setSelectedGender(selectedGender === "남" ? null : "남")}
              >
                남
              </Button>
              <Button
                variant={selectedGender === "여" ? "default" : "outline"}
                onClick={() => setSelectedGender(selectedGender === "여" ? null : "여")}
              >
                여
              </Button>
            </div>
            <div className="flex items-center space-x-4">
              <Label htmlFor="weight">체중 & 키</Label>
              <Input
                id="weight"
                placeholder="00 Kg"
                className="w-24"
                value={weight}
                onChange={(e) => setWeight(e.target.value)} // 체중 상태 업데이트
              />
              <Input
                id="height"
                placeholder="00 CM"
                className="w-24"
                value={height}
                onChange={(e) => setHeight(e.target.value)} // 키 상태 업데이트
              />
            </div>
            <div className="flex items-center space-x-4">
              <Label htmlFor="age">연령</Label>
              <Input
                id="age"
                placeholder="00 세"
                className="w-24"
                value={age}
                onChange={(e) => setAge(e.target.value)} // 연령 상태 업데이트
              />
            </div>
            <div className="space-y-2">
              <Label>질병&건강 상태</Label>
              <div className="flex flex-wrap gap-2">
                {["당뇨", "고혈압", "고지혈증", "관절염", "빈혈", "심혈관", "우울증", "감소성", "소화불량", "알레르기"].map(condition => (
                  <Button
                    key={condition}
                    variant={selectedHealthConditions.includes(condition) ? "default" : "outline"}
                    onClick={() => toggleSelection("health", condition)}
                  >
                    {condition}
                  </Button>
                ))}
                <Button variant="default">추가 +</Button>
              </div>
            </div>
            <div className="space-y-2">
              <Label>복용중인 영양제&보충제</Label>
              <div className="flex flex-wrap gap-2">
                {["종합비타민", "비타민C", "비타민A", "단백질", "아르기닌", "마그네슘", "철분", "여인"].map(supplement => (
                  <Button
                    key={supplement}
                    variant={selectedSupplements.includes(supplement) ? "default" : "outline"}
                    onClick={() => toggleSelection("supplements", supplement)}
                  >
                    {supplement}
                  </Button>
                ))}
                <Button variant="default">추가 +</Button>
              </div>
            </div>
            <div className="space-y-2">
              <Label>특이 사항</Label>
              <div className="flex flex-wrap gap-2">
                {["임신부", "수유중", "운동선수", "비건"].map(note => (
                  <Button
                    key={note}
                    variant={selectedSpecialNotes.includes(note) ? "default" : "outline"}
                    onClick={() => toggleSelection("specialNotes", note)}
                  >
                    {note}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>
      <footer className="flex justify-center w-full mt-8">
        <div className="flex space-x-4">
          <Link href="#" className="text-muted-foreground" prefetch={false}>
            FAQ
          </Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>
            Terms
          </Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>
            AI Policy
          </Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>
            Privacy
          </Link>
        </div>
      </footer>
    </div>
  );
}

function ArrowRightIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5 12h14" />
      <path d="m12 5 7 7-7 7" />
    </svg>
  );
}

function ChevronDownIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
