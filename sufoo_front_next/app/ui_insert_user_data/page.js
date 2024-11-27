"use client";

import { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import { useRouter } from "next/navigation";
import LoadingModal from "@/components/LoadingModal";

const CategorySection = ({ title, options, selectedItems, onToggle, category }) => {
  const [isAdding, setIsAdding] = useState(false);
  const [newItem, setNewItem] = useState("");

  const handleAddClick = () => setIsAdding(true);

  const handleInputSubmit = () => {
    if (newItem.trim()) {
      const newOption = { id: Date.now(), name: newItem.trim() };
      options.setter(prev => [...prev, newOption]);
      onToggle(category, newOption.name);
    }
    setIsAdding(false);
    setNewItem("");
  };

  return (
    <div className="space-y-2 bg-white p-4 rounded-lg shadow">
      <Label className="text-lg font-semibold">{title}</Label>
      <div className="flex flex-wrap gap-2">
        {options.value.map(option => (
          <Button
            key={option.id}
            variant={selectedItems.includes(option.name) ? "default" : "outline"}
            onClick={() => onToggle(category, option.name)}
            className="rounded-full"
          >
            {option.name}
          </Button>
        ))}
        <Button variant="outline" onClick={handleAddClick} className="rounded-full">
          추가 +
        </Button>
      </div>
      {isAdding && (
        <div className="mt-2 flex items-center space-x-2">
          <Input
            type="text"
            value={newItem}
            placeholder="새 항목 입력"
            onChange={(e) => setNewItem(e.target.value)}
            className="flex-grow"
          />
          <Button variant="default" onClick={handleInputSubmit}>
            완료
          </Button>
        </div>
      )}
    </div>
  );
};

export default function Component() {
  const router = useRouter();
  const [sessionId, setSessionId] = useState('');
  const [selectedGender, setSelectedGender] = useState(null);
  const [selectedHealthConditions, setSelectedHealthConditions] = useState([]);
  const [selectedSupplements, setSelectedSupplements] = useState([]);
  const [selectedSpecialNotes, setSelectedSpecialNotes] = useState([]);
  const [selectedDrugs, setSelectedDrugs] = useState([]);
  const [llmJsonData, setLlmJsonData] = useState('');
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

  const [healthOptions, setHealthOptions] = useState([
    { id: 1, name: '고혈압' },
    { id: 2, name: '당뇨병' },
  ]);
  const [supplementOptions, setSupplementOptions] = useState([
    { id: 1, name: '비타민 C' },
    { id: 2, name: '오메가3' },
  ]);
  const [drugOptions, setDrugOptions] = useState([
    { id: 1, name: '아스피린' },
    { id: 2, name: '메트포르민' },
  ]);
  const [specialOptions, setSpecialOptions] = useState([
    { id: 1, name: '임신' },
    { id: 2, name: '알레르기' },
  ]);

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

    console.log("Session ID: ", sessionId);

    const userData = {
      session_id: sessionId,
      gender: selectedGender,
      weight,
      height,
      age,
      searchTerm,
      healthConditions: selectedHealthConditions,
      supplements: selectedSupplements,
      specialNotes: selectedSpecialNotes,
      drugs: selectedDrugs,
    };

    const body = {
      userData,
      healthIds,
      drugIds,
      supplementIds,
      specialIds
    };

     // ui_loading 페이지로 이동
    router.push(`/ui_loading?question=${encodeURIComponent(searchTerm)}`);

    
    try {
      console.log("Session ID111: ", sessionId);      
      const res = await fetch('/api/insert_user_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      console.log("Session ID222: ", sessionId);

      const data = await res.json();
      if (res.ok) {
        setMessage(`사용자 정보가 성공적으로 저장되었습니다! 사용자 ID: ${data.userId}`);

        const llmJson = createLlmJson(userData, healthIds, drugIds, supplementIds, specialIds);
        setLlmJsonData(JSON.stringify(llmJson, null, 2));

        const randomPageId = generateRandomString(10);

        const query = new URLSearchParams({
          userId: String(data.userId),
          sessionId: String(sessionId),
          relative_url: String(randomPageId),
          llmJsonData: JSON.stringify(llmJson)
        }).toString();
        const url = `/content_page/${randomPageId}?${query}`;
        router.push(url);

      } else {
        setMessage(`에러 발생: ${data.message}`);
        console.log("Session ID: ", sessionId);
      }
    } catch (error) {
      setMessage(`API 호출 실패: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

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

  const generateRandomString = (length) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  };

  useEffect(() => {
    fetchSessionId();
  }, []);

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
      setSessionId('temp_' + Date.now().toString());
    }
  };

  return (
    <div className="flex flex-col items-center w-full min-h-screen p-4 bg-gray-100">
      <header className="flex items-center w-full px-4 py-2">
        <h1 className="text-lg font-bold">SUFOO Logo</h1>
      </header>
      <main className="w-full max-w-2xl space-y-6">
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
              className="w-full p-2 bg-gray-100 rounded"
            />
          </div>
          <CategorySection
            title="질병 & 건강 상태"
            options={{ value: healthOptions, setter: setHealthOptions }}
            selectedItems={selectedHealthConditions}
            onToggle={toggleSelection}
            category="health"
          />
          <CategorySection
            title="복용중인 영양제 & 보충제"
            options={{ value: supplementOptions, setter: setSupplementOptions }}
            selectedItems={selectedSupplements}
            onToggle={toggleSelection}
            category="supplements"
          />
          <CategorySection
            title="복용중인 약물"
            options={{ value: drugOptions, setter: setDrugOptions }}
            selectedItems={selectedDrugs}
            onToggle={toggleSelection}
            category="drugs"
          />
          <CategorySection
            title="특이 사항"
            options={{ value: specialOptions, setter: setSpecialOptions }}
            selectedItems={selectedSpecialNotes}
            onToggle={toggleSelection}
            category="specialNotes"
          />
        </section>
        {message && <p className="text-red-500 mt-4">{message}</p>}
        {loading && <LoadingModal />}
        {llmJsonData && (
          <div className="mt-4 p-4 bg-white rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-2">LLM JSON 데이터:</h3>
            <pre className="bg-gray-100 p-2 rounded overflow-x-auto">{llmJsonData}</pre>
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