'use client';

import { useAuth } from '@/context/AuthContext';
import LoginPage from './(auth)/login/page';
import dynamic from 'next/dynamic';
import AppShellSkeleton from '@/components/skeletons/AppShellSkeleton';

const AuthenticatedDashboard = dynamic(() => import('@/components/AuthenticatedDashboard'), {
  loading: () => <AppShellSkeleton />,
  ssr: false,
});

export default function Home() {
  const { user, isLoading } = useAuth();

  if (isLoading) return <AppShellSkeleton />;
  if (!user) return <LoginPage />;
  return <AuthenticatedDashboard />;
}
