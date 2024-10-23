"use client"; // 클라이언트 컴포넌트로 지정

import { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation"; 
import LoadingModal from "@/components/LoadingModal"; // 로딩 모달 컴포넌트 임포트

export default function Component() {
  const router = useRouter();
  const [sessionId, setSessionId] = useState('');
  const [selectedGender, setSelectedGender] = useState(null);
  const [selectedHealthConditions, setSelectedHealthConditions] = useState([]);
  const [selectedSupplements, setSelectedSupplements] = useState([]);
  const [selectedSpecialNotes, setSelectedSpecialNotes] = useState([]);
  const [selectedDrugs, setSelectedDrugs] = useState([]);
  const [llmJsonData, setLlmJsonData] = useState(''); // LLM JSON 데이터를 위한 새로운 상태
  const [weight, setWeight] = useState(""); 
  const [height, setHeight] = useState(""); 
  const [age, setAge] = useState(""); 
  const [searchTerm, setSearchTerm] = useState(""); 
  const [healthIds, setHealthIds] = useState([]);
  const [drugIds, setDrugIds] = useState([]);
  const [supplementIds, setSupplementIds] = useState([]);
  const [specialIds, setSpecialIds] = useState([]);
  const [loading, setLoading] = useState(false); 
  const [message, setMessage] = useState('');

  const healthOptions = [
    { id: 1, name: '고혈압' },
    { id: 2, name: '당뇨병' },
    // 추가 옵션
  ];

  const supplementOptions = [
    { id: 1, name: '비타민 C' },
    { id: 2, name: '오메가3' },
    // 추가 옵션
  ];

  const drugOptions = [
    { id: 1, name: '아스피린' },
    { id: 2, name: '메트포르민' },
    // 추가 옵션
  ];

  const specialOptions = [
    { id: 1, name: '임신' },
    { id: 2, name: '알레르기' },
    // 추가 옵션
  ];

  const toggleSelection = (category, item) => {
    const setSelectedFunction = {
      health: setSelectedHealthConditions,
      supplements: setSelectedSupplements,
      specialNotes: setSelectedSpecialNotes,
      drugs: setSelectedDrugs,
    };

    setSelectedFunction[category]((prevSelected) =>
      prevSelected.includes(item)
        ? prevSelected.filter((i) => i !== item)
        : [...prevSelected, item]
    );
  };

  const handleSubmit = async () => {
    setLoading(true);
    setMessage('');

    console.log("Session ID: ", sessionId); // 디버깅용 로그 추가

    const userData = {
      session_id : sessionId,
      gender: selectedGender,
      weight,
      height,
      age,
      searchTerm,
      healthConditions: selectedHealthConditions,
      supplements: selectedSupplements,
      specialNotes: selectedSpecialNotes,      
      drugs : selectedDrugs,
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
        //router.push('/ui_loading');
        console.log("Session ID111: ", sessionId); // 디버깅용 로그 추가
        const res = await fetch('/api/insert_user_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        console.log("Session ID222: ", sessionId); // 디버깅용 로그 추가

      const data = await res.json();
      if (res.ok) {
        setMessage(`사용자 정보가 성공적으로 저장되었습니다! 사용자 ID: ${data.userId}`);

        // LLM을 위한 JSON 데이터 생성
        const llmJson = createLlmJson(userData, healthIds, drugIds, supplementIds, specialIds);
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
        console.log("Session ID: ", sessionId); // 디버깅용 로그 추가
        //alert('에러가 발생했습니다. 다시 검색 부탁드립니다.');
        //router.back(); // 이전 페이지로 돌아감
        
        //const url = `/ui_loading`;
        // 페이지 이동
        //router.push(url);

    }
  } catch (error) {
      setMessage(`API 호출 실패: ${error.message}`);
      //alert('에러가 발생했습니다. 다시 검색 부탁드립니다.');
      //router.back(); // 이전 페이지로 돌아감

      //const url = `/ui_loading`;
      // 페이지 이동
      //router.push(url);
  } finally {
      setLoading(false);
  }
  };

   // LLM JSON 데이터 생성 함수
   const createLlmJson = (userData, healthIds, drugIds, supplementIds, specialIds) => {
    const getNames = (ids, options) => ids.map(id => options.find(opt => opt.id === id)?.name || '');

    return {
        question: searchTerm,
        additional_info_for_question: "",
        client_info: {
            gender: userData.gender,
            weight: userData.weight.toString(),
            height: userData.height.toString(),
            bmi: (userData.weight / Math.pow(userData.height / 100, 2)).toFixed(1),
            health_conditions: getNames(healthIds, healthOptions),
            medications_being_taken: getNames(drugIds, drugOptions),
            supplements_being_taken: getNames(supplementIds, supplementOptions),
            special_conditions: getNames(specialIds, specialOptions)
        },
        request: [
            {
                title: "",
                description: "",
                result: "",
                subject: [
                    { sub_title: "", sub_description: "", sub_result: "" },
                    { sub_title: "", sub_description: "", sub_result: "" }
                ]
            },
            {
                title: "",
                description: "",
                result: "",
                subject: [
                    { sub_title: "", sub_description: "", sub_result: "" },
                    { sub_title: "", sub_description: "", sub_result: "" }
                ]
            }
        ]
    };
};

// 랜덤 문자열 생성 함수 추가
const generateRandomString = (length) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
};

useEffect(() => {
    // 컴포넌트가 마운트될 때 세션 ID를 가져옵니다.
    fetchSessionId();
}, [router]);

const fetchSessionId = async () => {
    try {
        const response = await fetch('/api/get_session_id');
        if (!response.ok) {
            throw new Error('서버 응답이 올바르지 않습니다.');
        }
        const data = await response.json();
        setSessionId(data.sessionId);
    } catch (error) {
        console.error('세션 ID를 가져오는 데 실패했습니다:', error);
        // 에러 발생 시 임시 ID 생성
        setSessionId('temp_' + Date.now().toString());
    }
};

  return (
    <div className="flex flex-col items-center w-full min-h-screen p-4">
      <header className="flex items-center w-full px-4 py-2">
        <h1 className="text-lg font-bold">SUFOO Logo</h1>
      </header>
      <main className="w-full max-w-2xl">
        <section className="text-center">
          <h2 className="text-3xl font-bold">당신의 건강을 위한 영양 검색!</h2>
          <p className="text-muted-foreground">설명영역설명12315554</p>
        </section>
        <section className="mt-4">
          <Input
            type="search"
            placeholder="당신에게 좋은 음식은?"
            className="w-full pl-8 pr-12 py-2"
            value={searchTerm} 
            //question
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <div className="flex justify-between mt-2">
            <Button variant="outline" onClick={handleSubmit} disabled={loading}>
              {loading ? '저장 중...' : '데이터 저장'}
            </Button>
            <Button variant="outline">상세 검색</Button>
          </div>
        </section>
        <section className="mt-4 space-y-4">
          <div className="flex items-center space-x-4">
            <Label htmlFor="gender">성별</Label>
            <Button variant={selectedGender === "남" ? "default" : "outline"} onClick={() => setSelectedGender(selectedGender === "남" ? null : "남")}>남</Button>
            <Button variant={selectedGender === "여" ? "default" : "outline"} onClick={() => setSelectedGender(selectedGender === "여" ? null : "여")}>여</Button>
          </div>
          <div className="flex items-center space-x-4">
            <Label htmlFor="weight">체중 & 키</Label>
            <Input id="weight" placeholder="00 Kg" className="w-24" value={weight} onChange={(e) => setWeight(e.target.value)} />
            <Input id="height" placeholder="00 CM" className="w-24" value={height} onChange={(e) => setHeight(e.target.value)} />
          </div>
          <div className="flex items-center space-x-4">
            <Label htmlFor="age">연령</Label>
            <Input id="age" placeholder="00 세" className="w-24" value={age} onChange={(e) => setAge(e.target.value)} />
          </div>
          <div className="space-y-2">
                <label>세션 ID:</label>
                <input
                    type="text"
                    value={sessionId}
                    readOnly
                    //style={{...styles.input, backgroundColor: '#f0f0f0'}}
                />
          </div>
          <div className="space-y-2">
            <Label>질병&건강 상태</Label>
            <div className="flex flex-wrap gap-2">
                {healthOptions.map(option => (
                  <Button
                    key={option.id}
                    variant={selectedHealthConditions.includes(option.name) ? "default" : "outline"}
                    onClick={() => toggleSelection("health",option.name)}
                  >
                    {option.name}
                  </Button>
             ))}
             <Button variant="default">추가 +</Button>
            </div>
          </div>
          <div className="space-y-2">
            <Label>복용중인 영양제&보충제</Label>
            <div className="flex flex-wrap gap-2">
                {supplementOptions.map(option => (
                  <Button
                    key={option.id}
                    variant={selectedSupplements.includes(option.name) ? "default" : "outline"}
                    onClick={() => toggleSelection("supplements",option.name)}
                  >
                    {option.name}
                  </Button>
             ))}
             <Button variant="default">추가 +</Button>
            </div>
          </div>
          <div className="space-y-2">
            <Label>복용중인 약물</Label>
            <div className="flex flex-wrap gap-2">
                {drugOptions.map(option => (
                  <Button
                    key={option.id}
                    variant={selectedDrugs.includes(option.name) ? "default" : "outline"}
                    onClick={() => toggleSelection("drugs",option.name)}
                  >
                    {option.name}
                  </Button>
             ))}
             <Button variant="default">추가 +</Button>
            </div>
          </div>
          <div className="space-y-2">
            <Label>특이 사항</Label>
            <div className="flex flex-wrap gap-2">
                {specialOptions.map(option => (
                  <Button
                    key={option.id}
                    variant={selectedSpecialNotes.includes(option.name) ? "default" : "outline"}
                    onClick={() => toggleSelection("specialNotes",option.name)}
                  >
                    {option.name}
                  </Button>
             ))}
             <Button variant="default">추가 +</Button>
            </div>
          </div>
        </section>
        {message && <p className="text-red-500 mt-4">{message}</p>} {/* 오류 메시지 표시 */}

        {/* {loading && (
          <div className="fixed inset-0 flex items-center justify-center bg-gray-500 bg-opacity-50 z-50">
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <p className="text-lg font-bold">로딩 중...</p>
            </div>
          </div>
        )} */}
        {/* 로딩 모달창 표시 */}
        {loading && <LoadingModal />}

        {llmJsonData && (
                <div style={styles.section}>
                    <h3>LLM JSON 데이터:</h3>
                    <pre style={styles.jsonDisplay}>{llmJsonData}</pre>
                </div>
         )}
      </main>
      <footer className="flex justify-center w-full mt-8">
        <div className="flex space-x-4">
          <Link href="#" className="text-muted-foreground" prefetch={false}>FAQ</Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>Terms</Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>AI Policy</Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>Privacy</Link>
        </div>
      </footer>
    </div>
  );
}

// 스타일 정의
const styles = {
  container: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      maxWidth: '600px',
      margin: '0 auto',
      padding: '20px',
      backgroundColor: '#f9f9f9',
      borderRadius: '8px',
      boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.1)',
  },
  section: {
      marginBottom: '15px',
      width: '100%',
      padding: '10px',
      border: '1px solid #ddd',
      borderRadius: '5px',
      backgroundColor: '#fff',
  },
  input: {
      width: '100%',
      padding: '10px',
      marginTop: '5px',
      borderRadius: '5px',
      border: '1px solid #ccc',
  },
  button: {
      padding: '10px 20px',
      backgroundColor: '#0070f3',
      color: '#fff',
      border: 'none',
      borderRadius: '5px',
      cursor: 'pointer',
      marginTop: '20px',
      width: '100%',
  },
  message: {
      marginTop: '20px',
      padding: '10px',
      border: '1px solid',
      borderRadius: '5px',
      width: '100%',
      textAlign: 'center',
  },
  jsonDisplay: {
      backgroundColor: '#f0f0f0',
      padding: '10px',
      borderRadius: '5px',
      overflowX: 'auto',
      whiteSpace: 'pre-wrap',
      wordWrap: 'break-word'
  },
};
