"use client";

import { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import { useRouter } from "next/navigation";
import LoadingModal from "@/components/LoadingModal";
import { FaChevronUp, FaChevronDown } from 'react-icons/fa';

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
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false); // 상세 검색 초기 상태: 접혀있음


  // // DB에서 불러오기 (느려서 중지)
  // const [healthOptions, setHealthOptions] = useState([]);
  // const [supplementOptions, setSupplementOptions] = useState([]);
  // const [drugOptions, setDrugOptions] = useState([]);
  // const [specialOptions, setSpecialOptions] = useState([]);

  // useEffect(() => {
  //   fetchConditions();
  // }, []);

  const [healthOptions, setHealthOptions] = useState([
    { id: 1, name: '고혈압', name_en: 'Hypertension' },
    { id: 2, name: '당뇨', name_en: 'Diabetes' },
    { id: 3, name: '고지혈증', name_en: 'Hyperlipidemia' },
    { id: 4, name: '비만', name_en: 'Obesity' },
    { id: 5, name: '천식', name_en: 'Asthma' },
    { id: 6, name: '알레르기', name_en: 'Allergies' },
    { id: 7, name: '심장 질환', name_en: 'Heart Disease' },
    { id: 8, name: '암', name_en: 'Cancer' },
    { id: 9, name: '우울증', name_en: 'Depression' },
    { id: 10, name: '불안 장애', name_en: 'Anxiety Disorder' },
    { id: 11, name: '수면 장애', name_en: 'Sleep Disorder' },
    { id: 12, name: '관절염', name_en: 'Arthritis' },
    { id: 13, name: '소화 장애', name_en: 'Digestive Issues' },
    { id: 14, name: '만성 피로 증후군', name_en: 'Chronic Fatigue Syndrome' },
    { id: 15, name: '편두통', name_en: 'Migraine' },
    { id: 16, name: '갑상선 질환', name_en: 'Thyroid Disorder' },
    { id: 17, name: '신장 질환', name_en: 'Kidney Disease' },
    { id: 18, name: '간 질환', name_en: 'Liver Disease' },
  ]);
  const [supplementOptions, setSupplementOptions] = useState([
    { id: 1, name: '종합 비타민', name_en: 'Multivitamins' },
    { id: 2, name: '비타민 C', name_en: 'Vitamin C' },
    { id: 3, name: '비타민 D', name_en: 'Vitamin D' },
    { id: 4, name: '오메가-3', name_en: 'Omega-3' },
    { id: 5, name: '프로바이오틱스', name_en: 'Probiotics' },
    { id: 6, name: '칼슘', name_en: 'Calcium' },
    { id: 7, name: '마그네슘', name_en: 'Magnesium' },
    { id: 8, name: '아연', name_en: 'Zinc' },
    { id: 9, name: '철분', name_en: 'Iron' },
    { id: 10, name: '비타민 B 복합체', name_en: 'Vitamin B Complex' },
    { id: 11, name: '콜라겐', name_en: 'Collagen' },
    { id: 12, name: '단백질 보충제', name_en: 'Protein Supplements' },
    { id: 13, name: '글루코사민', name_en: 'Glucosamine' },
    { id: 14, name: '커큐민', name_en: 'Curcumin' },
    { id: 15, name: '엽산', name_en: 'Folic Acid' },
    { id: 16, name: '코엔자임 Q10', name_en: 'Coenzyme Q10' },
    { id: 17, name: '멜라토닌', name_en: 'Melatonin' },
  ]);
  const [drugOptions, setDrugOptions] = useState([
    { id: 1, name: '고혈압 약', name_en: 'Hypertension medication' },
    { id: 2, name: '당뇨 약', name_en: 'Diabetes medication' },
    { id: 3, name: '고지혈증 약', name_en: 'Hyperlipidemia medication' },
    { id: 4, name: '비만 치료제', name_en: 'Obesity medication' },
    { id: 5, name: '천식 약', name_en: 'Asthma medication' },
    { id: 6, name: '알레르기 약', name_en: 'Allergy medication' },
    { id: 7, name: '심장 질환 약', name_en: 'Heart Disease medication' },
    { id: 8, name: '암 치료제', name_en: 'Cancer treatment' },
    { id: 9, name: '우울증 약', name_en: 'Depression medication' },
    { id: 10, name: '불안 장애 약', name_en: 'Anxiety medication' },
    { id: 11, name: '수면제', name_en: 'Sleep medication' },
    { id: 12, name: '관절염 약', name_en: 'Arthritis medication' },
    { id: 13, name: '소화제', name_en: 'Digestive medication' },
  ]);
  const [specialOptions, setSpecialOptions] = useState([
    { id: 1, name: '임산부', name_en: 'Pregnant' },
    { id: 2, name: '수유 중', name_en: 'Breastfeeding' },
    { id: 3, name: '운동선수', name_en: 'Athlete' },
    { id: 4, name: '채식주의자', name_en: 'Vegetarian' },
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

    const userData = {
      session_id: sessionId,
      searchTerm,
      gender: selectedGender || '여',  // 기본값을 '남'으로 설정
      weight: weight || '0',  // 기본값을 '0'으로 설정
      height: height || '0',  // 기본값을 '0'으로 설정
      age: age || '0' // 기본값을 '0'으로 설정
    };

    // if (selectedGender) userData.gender = selectedGender;
    // if (weight) userData.weight = weight;
    // if (height) userData.height = height;
    // if (age) userData.age = age;

    if (!searchTerm) {
      setMessage('검색어를 입력해주세요.'); // 검색어가 없을 때 메시지 표시
      setLoading(false);
      return;
    }

    // ui_loading 페이지로 이동
    router.push(`/ui_loading?question=${encodeURIComponent(searchTerm)}`);

    const body = {
      userData,
      healthIds,
      drugIds,
      supplementIds,
      specialIds
    };

    


    try {
      const res = await fetch('/api/insert_user_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      
      const data = await res.json();
      if (res.ok) {
        // setMessage(`사용자 정보가 성공적으로 저장되었습니다! 사용자 ID: ${data.userId}`);

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


  const fetchConditions = async () => {
    try {
      const res = await fetch('/api/select_conditions'); // API endpoint 호출
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();

      setHealthOptions(data.healthConditions.map((item, index) => ({ id: index + 1, name: item.health_name })));
      setSupplementOptions(data.supplements.map((item, index) => ({ id: index + 1, name: item.supplement_name })));
      setDrugOptions(data.drugs.map((item, index) => ({ id: index + 1, name: item.drug_name })));
      setSpecialOptions(data.specialConditions.map((item, index) => ({ id: index + 1, name: item.special_name })));

    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };
  
  const toggleAdvancedSearch = () => {
    setShowAdvancedSearch(!showAdvancedSearch);
  };

  return (
    <div className={`flex flex-col items-center w-full min-h-screen p-4 bg-gray-100`}>
      <header className="flex items-center w-full px-2 py-1">
        <img
          src="/logo/logo_main_0.png"
          alt="FODOIT Logo"
          className="h-4 sm:h-6 md:h-8 lg:h-8"
        />
      </header>
      <main className={`w-full max-w-2xl space-y-6 flex-grow flex flex-col items-center transition-all duration-300 ${showAdvancedSearch ? 'items-start' : 'justify-center'}`}>
        <section className="text-center ibm-plex-sans-kr-regular">
          <h2 className="text-2xl sm:text-2xl md:text-4xl font-bold mb-4">내 몸에 꼭 맞는 건강 정보</h2>
          <h3 className="text-md sm:text-md md:text-2xl mb-2">인공지능으로 나의 체질, 질병, 복용 약물까지 고려하여 맞춤형 분석 정보 제공</h3>
        </section>
        <section className="mt-4 w-full">
          <Input
            type="search"
            placeholder="당신에게 좋은 음식은?"
            className="w-full pl-8 pr-12 py-2"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleSubmit();
              }
            }}
          />
          <div className="mt-2 w-full flex flex-col items-end space-y-2"> {/* flex-col and items-end for vertical alignment */}
          <Button variant="outline" onClick={handleSubmit} disabled={loading}>
            {loading ? '저장 중...' : '검색'}
          </Button>
          <Button variant="ghost" onClick={toggleAdvancedSearch} className="flex items-center">
            {showAdvancedSearch ? <FaChevronUp className="mr-2" /> : <FaChevronDown className="mr-2" />}
            상세 검색
          </Button>
        </div>
        </section>
         {showAdvancedSearch && (
           <section className="mt-4 space-y-4 w-full">
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
                onChange={(e) => setWeight(e.target.value)}
              />
              <Input
                id="height"
                placeholder="00 CM"
                className="w-24"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
              />
            </div>
            <div className="flex items-center space-x-4">
              <Label htmlFor="age">연령</Label>
              <Input
                id="age"
                placeholder="00 세"
                className="w-24"
                value={age}
                onChange={(e) => setAge(e.target.value)}
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
          )}
        {message && <p className="text-red-500 mt-4">{message}</p>}
        {/* {loading && <LoadingModal />} */}
        {/* {llmJsonData && (
          <div className="mt-4 p-4 bg-white rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-2">LLM JSON 데이터:</h3>
            <pre className="bg-gray-100 p-2 rounded overflow-x-auto">{llmJsonData}</pre>
          </div>
        )} */}
    
      </main>

      <div className="mt-8 max-w-2xl w-full mx-auto"> {/* max-w-2xl와 mx-auto 추가 */}
        <iframe 
          src="https://ads-partners.coupang.com/widgets.html?id=822765&template=carousel&trackingCode=AF7114013&subId=&width=680&height=140&tsource=" 
          width="100%" 
          height="140" 
          frameBorder="0" 
          scrolling="no" 
          referrerPolicy="unsafe-url" 
          browsingtopics="true">
        </iframe>
        <iframe 
          src="https://coupa.ng/cg0JDb" 
          width="100%" 
          height="36" 
          frameBorder="0" 
          scrolling="no" 
          referrerPolicy="unsafe-url" 
          browsingtopics="true">
        </iframe>
      </div>
      
      {/* <footer className="flex justify-center w-full mt-8">
        <div className="flex space-x-4">
          <Link href="#" className="text-muted-foreground" prefetch={false}>FAQ</Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>Terms</Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>AI Policy</Link>
          <Link href="#" className="text-muted-foreground" prefetch={false}>Privacy</Link>
        </div>
      </footer> */}
    </div>
  );
}