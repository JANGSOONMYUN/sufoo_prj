"use client";

import { useState, useEffect, useMemo } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useRouter } from "next/navigation";
import { FaChevronUp, FaChevronDown } from 'react-icons/fa';

const ADV_SEARCH_PERSIST_COOKIE_NAME = "sufoo_adv_search_v1";
const ADV_SEARCH_PERSIST_COOKIE_MAX_AGE_SEC = 60 * 60 * 24; // 1일

const safeJsonParse = (str) => {
  try {
    return JSON.parse(str);
  } catch (e) {
    return null;
  }
};

const readCookie = (name) => {
  if (typeof document === "undefined") return null;
  const cookieStr = document.cookie || "";
  const parts = cookieStr.split("; ").filter(Boolean);
  const found = parts.find((p) => p.startsWith(`${name}=`));
  if (!found) return null;
  return decodeURIComponent(found.slice(name.length + 1));
};

const writeCookie = (name, value, maxAgeSeconds) => {
  if (typeof document === "undefined") return;
  const secure = typeof window !== "undefined" && window.location?.protocol === "https:" ? "; secure" : "";
  document.cookie = `${name}=${encodeURIComponent(value)}; max-age=${maxAgeSeconds}; path=/; samesite=lax${secure}`;
};

const CategorySection = ({ title, options, selectedItems, onToggle, category, onAddCustom }) => {
  const [isAdding, setIsAdding] = useState(false);
  const [newItem, setNewItem] = useState("");

  const handleAddClick = () => setIsAdding(true);

  const handleInputSubmit = () => {
    const name = newItem.trim();
    if (name) {
      const exists = Array.isArray(options?.value) && options.value.some((opt) => opt?.name === name);
      if (exists) {
        onToggle(category, name);
      } else {
        const newOption = { id: Date.now(), name, name_en: name, is_custom: true };
        options.setter((prev) => [...prev, newOption]);
        onAddCustom?.(category, name);
        onToggle(category, name);
      }
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
  const [weight, setWeight] = useState("");
  const [height, setHeight] = useState("");
  const [age, setAge] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false); // 상세 검색 초기 상태: 접혀있음
  const [recommendations, setRecommendations] = useState([]);
  const [customOptionsByCategory, setCustomOptionsByCategory] = useState({
    health: [],
    supplements: [],
    drugs: [],
    specialNotes: [],
  });
  const [persistReady, setPersistReady] = useState(false);


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
    { id: 5, name: '흡연', name_en: 'Smoking' },
  ]);

  // DB 저장용 ID 배열(커스텀 항목은 제외)
  const healthIds = useMemo(() => {
    const ids = [];
    for (const name of selectedHealthConditions) {
      const opt = healthOptions.find((o) => o?.name === name && !o?.is_custom);
      if (typeof opt?.id === "number") ids.push(opt.id);
    }
    return ids;
  }, [selectedHealthConditions, healthOptions]);

  const supplementIds = useMemo(() => {
    const ids = [];
    for (const name of selectedSupplements) {
      const opt = supplementOptions.find((o) => o?.name === name && !o?.is_custom);
      if (typeof opt?.id === "number") ids.push(opt.id);
    }
    return ids;
  }, [selectedSupplements, supplementOptions]);

  const drugIds = useMemo(() => {
    const ids = [];
    for (const name of selectedDrugs) {
      const opt = drugOptions.find((o) => o?.name === name && !o?.is_custom);
      if (typeof opt?.id === "number") ids.push(opt.id);
    }
    return ids;
  }, [selectedDrugs, drugOptions]);

  const specialIds = useMemo(() => {
    const ids = [];
    for (const name of selectedSpecialNotes) {
      const opt = specialOptions.find((o) => o?.name === name && !o?.is_custom);
      if (typeof opt?.id === "number") ids.push(opt.id);
    }
    return ids;
  }, [selectedSpecialNotes, specialOptions]);

  const handleAddCustomOption = (category, name) => {
    setCustomOptionsByCategory((prev) => {
      const cur = Array.isArray(prev?.[category]) ? prev[category] : [];
      if (cur.includes(name)) return prev;
      return { ...prev, [category]: [...cur, name] };
    });
  };

  const mergeOptionsWithCustomNames = (prevOptions, customNames) => {
    if (!Array.isArray(customNames) || customNames.length === 0) return prevOptions;
    const existingNames = new Set((prevOptions || []).map((o) => o?.name).filter(Boolean));
    const toAdd = [];
    for (const raw of customNames) {
      const name = typeof raw === "string" ? raw.trim() : "";
      if (!name) continue;
      if (existingNames.has(name)) continue;
      existingNames.add(name);
      toAdd.push({ id: Date.now() + toAdd.length, name, name_en: name, is_custom: true });
    }
    return toAdd.length ? [...prevOptions, ...toAdd] : prevOptions;
  };

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
      gender: selectedGender || '알수없음',  // 기본값 선택안함
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

        const llmJson = createLlmJson(userData, {
          healthConditions: selectedHealthConditions,
          medicationsBeingTaken: selectedDrugs,
          supplementsBeingTaken: selectedSupplements,
          specialConditions: selectedSpecialNotes,
        });

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

  const createLlmJson = (userData, selections) => {
    const weightNum = Number(userData.weight);
    const heightNum = Number(userData.height);
    const bmi =
      Number.isFinite(weightNum) && Number.isFinite(heightNum) && heightNum > 0 && weightNum > 0
        ? (weightNum / Math.pow(heightNum / 100, 2)).toFixed(1)
        : "";

    return {
      question: searchTerm,
      additional_info_for_question: "",
      client_info: {
        gender: userData.gender,
        weight: userData.weight.toString(),
        height: userData.height.toString(),
        bmi,
        health_conditions: Array.isArray(selections?.healthConditions) ? selections.healthConditions : [],
        medications_being_taken: Array.isArray(selections?.medicationsBeingTaken) ? selections.medicationsBeingTaken : [],
        supplements_being_taken: Array.isArray(selections?.supplementsBeingTaken) ? selections.supplementsBeingTaken : [],
        special_conditions: Array.isArray(selections?.specialConditions) ? selections.specialConditions : [],
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
    fetchRecommend();
    // 상세 검색(선택값/추가 항목) 복원
    const raw = readCookie(ADV_SEARCH_PERSIST_COOKIE_NAME);
    const parsed = raw ? safeJsonParse(raw) : null;
    if (parsed && parsed.v === 1) {
      const userInfo = parsed.userInfo || {};
      const selected = parsed.selected || {};
      const custom = parsed.custom || {};

      if (typeof userInfo.gender === "string" && (userInfo.gender === "남" || userInfo.gender === "여")) {
        setSelectedGender(userInfo.gender);
      } else {
        setSelectedGender(null);
      }
      setWeight(typeof userInfo.weight === "string" ? userInfo.weight : "");
      setHeight(typeof userInfo.height === "string" ? userInfo.height : "");
      setAge(typeof userInfo.age === "string" ? userInfo.age : "");

      setSelectedHealthConditions(Array.isArray(selected.health) ? selected.health : []);
      setSelectedSupplements(Array.isArray(selected.supplements) ? selected.supplements : []);
      setSelectedDrugs(Array.isArray(selected.drugs) ? selected.drugs : []);
      setSelectedSpecialNotes(Array.isArray(selected.specialNotes) ? selected.specialNotes : []);

      setCustomOptionsByCategory({
        health: Array.isArray(custom.health) ? custom.health : [],
        supplements: Array.isArray(custom.supplements) ? custom.supplements : [],
        drugs: Array.isArray(custom.drugs) ? custom.drugs : [],
        specialNotes: Array.isArray(custom.specialNotes) ? custom.specialNotes : [],
      });

      setHealthOptions((prev) => mergeOptionsWithCustomNames(prev, custom.health));
      setSupplementOptions((prev) => mergeOptionsWithCustomNames(prev, custom.supplements));
      setDrugOptions((prev) => mergeOptionsWithCustomNames(prev, custom.drugs));
      setSpecialOptions((prev) => mergeOptionsWithCustomNames(prev, custom.specialNotes));
    }
    setPersistReady(true);
  }, []);

  // 상세 검색(선택값/추가 항목) 저장: 1일짜리 쿠키
  useEffect(() => {
    if (!persistReady) return;
    const payload = {
      v: 1,
      updatedAt: Date.now(),
      userInfo: {
        gender: selectedGender,
        weight,
        height,
        age,
      },
      selected: {
        health: selectedHealthConditions,
        supplements: selectedSupplements,
        drugs: selectedDrugs,
        specialNotes: selectedSpecialNotes,
      },
      custom: customOptionsByCategory,
    };
    writeCookie(ADV_SEARCH_PERSIST_COOKIE_NAME, JSON.stringify(payload), ADV_SEARCH_PERSIST_COOKIE_MAX_AGE_SEC);
  }, [
    persistReady,
    selectedGender,
    weight,
    height,
    age,
    selectedHealthConditions,
    selectedSupplements,
    selectedDrugs,
    selectedSpecialNotes,
    customOptionsByCategory,
  ]);

  const selectedTagLabels = useMemo(() => {
    const tags = [];
    if (selectedGender) tags.push(`성별:${selectedGender}`);
    if (weight) tags.push(`체중:${weight}kg`);
    if (height) tags.push(`키:${height}cm`);
    if (age) tags.push(`연령:${age}세`);

    for (const v of selectedHealthConditions) tags.push(v);
    for (const v of selectedSupplements) tags.push(v);
    for (const v of selectedDrugs) tags.push(v);
    for (const v of selectedSpecialNotes) tags.push(v);

    return [...new Set(tags.map((t) => String(t || "").trim()).filter(Boolean))];
  }, [
    selectedGender,
    weight,
    height,
    age,
    selectedHealthConditions,
    selectedSupplements,
    selectedDrugs,
    selectedSpecialNotes,
  ]);

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

  const fetchRecommend = async () => {
    try { 
      const res = await fetch('api/recommend' , {
        method : 'GET',
        headers : {
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache",
          "Expires": "0"
        }
      }); // API 호출
      if(!res.ok){
        throw new Error(`HTTP error! status!!!!####: ${res.status}`);
      }
      const data = await res.json();
      setRecommendations(data.recommendData.map(item => item.search_words));
    } catch (error) {
      console.log("추천 검색어를 가져오는 데 실패했습니다.",error);
      setRecommendations([]);
    }
  }


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

  const clearAdvancedSearchConditions = () => {
    setSelectedGender(null);
    setWeight("");
    setHeight("");
    setAge("");
    setSelectedHealthConditions([]);
    setSelectedSupplements([]);
    setSelectedDrugs([]);
    setSelectedSpecialNotes([]);
  };

  const handleSearchTermChange = (word) => {
    return new Promise((resolve) => {
      setSearchTerm(word);
      resolve();
    });
  };

  const handleClick = (word) => {
    handleSearchTermChange(word).then(() => {
      // searchTerm이 업데이트된 후에 handleSubmit 호출
      // handleSubmit();
    }).catch((error) => {
      console.error("Error during search term change:", error);
    });
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
            autoComplete="off"
            data-form-type="other"
            name="search-term"
          />

           {/* 추천 검색어 표시 부분 */}
          <div className="mt-4">
            <div className="flex flex-wrap gap-2 mt-2">
              {recommendations.length > 0 ? (
                recommendations.map((word, index) => (
                  <button
                    type="button"
                    key={index}
                    className="px-3 py-1.5 rounded-lg bg-sky-50 text-sky-800 border border-sky-200 hover:bg-sky-100 transition-colors"
                    onClick={() => handleClick(word)}  // 클릭 시 바로 handleSubmit 실행
                  >
                    {word}
                  </button>
                ))
              ) : (
                <span className="text-sm text-muted-foreground">추천 검색어가 없습니다.</span>
              )}
            </div>
          </div>

          <div className="mt-3 w-full flex flex-col items-end space-y-2"> {/* flex-col and items-end for vertical alignment */}
          {selectedTagLabels.length > 0 && (
            <div className="w-full">
              <Label className="block w-full text-right text-sm font-semibold text-fuchsia-800">선택된 조건</Label>
              <div className="mt-2 w-full flex flex-wrap gap-2 justify-end">
                {selectedTagLabels.map((t) => (
                  <span
                    key={t}
                    className="px-2.5 py-1 rounded-lg bg-fuchsia-50 text-fuchsia-800 text-xs"
                  >
                    #{t}
                  </span>
                ))}
              </div>
            </div>
          )}
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
            <div className="w-full flex justify-end">
              <Button variant="outline" size="sm" onClick={clearAdvancedSearchConditions}>
                조건 지우기
              </Button>
            </div>
            <div className="space-y-2 bg-white p-4 rounded-lg shadow">
              <Label className="text-lg font-semibold">기본 정보</Label>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm text-muted-foreground">성별</Label>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant={selectedGender === "남" ? "default" : "outline"}
                      onClick={() => setSelectedGender(selectedGender === "남" ? null : "남")}
                      className="rounded-full"
                    >
                      남
                    </Button>
                    <Button
                      variant={selectedGender === "여" ? "default" : "outline"}
                      onClick={() => setSelectedGender(selectedGender === "여" ? null : "여")}
                      className="rounded-full"
                    >
                      여
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm text-muted-foreground">체중(kg) / 키(cm)</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <Input
                      id="weight"
                      placeholder="Kg"
                      value={weight}
                      type="number"
                      onChange={(e) => setWeight(e.target.value)}
                    />
                    <Input
                      id="height"
                      placeholder="Cm"
                      value={height}
                      type="number"
                      onChange={(e) => setHeight(e.target.value)}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="age" className="text-sm text-muted-foreground">
                    연령(세)
                  </Label>
                  <Input
                    id="age"
                    placeholder="세"
                    type="number"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                  />
                </div>
              </div>
            </div>
            
              <CategorySection
                title="질병 & 건강 상태"
                options={{ value: healthOptions, setter: setHealthOptions }}
                selectedItems={selectedHealthConditions}
                onToggle={toggleSelection}
                category="health"
                onAddCustom={handleAddCustomOption}
              />
              <CategorySection
                title="복용중인 영양제 & 보충제"
                options={{ value: supplementOptions, setter: setSupplementOptions }}
                selectedItems={selectedSupplements}
                onToggle={toggleSelection}
                category="supplements"
                onAddCustom={handleAddCustomOption}
              />
              <CategorySection
                title="복용중인 약물"
                options={{ value: drugOptions, setter: setDrugOptions }}
                selectedItems={selectedDrugs}
                onToggle={toggleSelection}
                category="drugs"
                onAddCustom={handleAddCustomOption}
              />
              <CategorySection
                title="특이 사항"
                options={{ value: specialOptions, setter: setSpecialOptions }}
                selectedItems={selectedSpecialNotes}
                onToggle={toggleSelection}
                category="specialNotes"
                onAddCustom={handleAddCustomOption}
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
        <p className="text-xs space-y-2 content-style">쿠팡 파트너스 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받습니다.</p>
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