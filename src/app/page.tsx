'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.push('/get-started/introduction');
  }, [router]);

  return (
    <div className="min-h-screen bg-[#0D1117] text-[#E6EDF3] flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#58A6FF] mx-auto mb-4"></div>
        <p className="text-gray-400">Redirecting to documentation...</p>
      </div>
    </div>
  );
}
