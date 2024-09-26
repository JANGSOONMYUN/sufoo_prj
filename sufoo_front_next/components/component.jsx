"use client"; // 클라이언트 컴포넌트로 지정

import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import Link from "next/link"

export default function Component() {
  const [selectedGender, setSelectedGender] = useState(null);
  const [selectedHealthConditions, setSelectedHealthConditions] = useState([]);
  const [selectedSupplements, setSelectedSupplements] = useState([]);
  const [selectedSpecialNotes, setSelectedSpecialNotes] = useState([]);

  const toggleSelection = (category, item) => {
    const setSelectedFunction = {
      health: setSelectedHealthConditions,
      supplements: setSelectedSupplements,
      specialNotes: setSelectedSpecialNotes
    }
    
    const selectedSet = {
      health: selectedHealthConditions,
      supplements: selectedSupplements,
      specialNotes: selectedSpecialNotes
    }
    
    setSelectedFunction[category](prevSelected => (
      prevSelected.includes(item) 
        ? prevSelected.filter(i => i !== item) 
        : [...prevSelected, item]
    ))
  }

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
          <p className="text-muted-foreground">설명영역설명</p>
        </section>
        <section className="mt-4">
          <div className="relative">
            <Input type="search" placeholder="당신에게 좋은 음식은?" className="w-full pl-8 pr-12 py-2" />
            <ArrowRightIcon className="absolute right-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          </div>
          <div className="flex justify-between mt-2">
            <div className="flex space-x-2">
              <Button variant="outline">당신에게 좋은 음식은?</Button>
              <Button variant="outline">당신에게 좋은 음식은?</Button>
              <Button variant="outline">당신에게 좋은 음식은?</Button>
            </div>
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
              <Input id="weight" placeholder="00 Kg" className="w-24" />
              <Input id="height" placeholder="00 CM" className="w-24" />
            </div>
            <div className="flex items-center space-x-4">
              <Label htmlFor="age">연령</Label>
              <Input id="age" placeholder="00 세" className="w-24" />
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
  )
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
  )
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
  )
}