import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ToolBrain - Flexible Reinforcement Learning Framework for Agentic Systems",
  description: "Go from a failing, unreliable agent to a proficient, tool-using expert. ToolBrain makes the power of RL for agents Easy-to-Use, Flexible, and Efficient.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#0D1117] text-[#E6EDF3]`}
      >
        {children}
      </body>
    </html>
  );
}
