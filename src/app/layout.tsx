import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import SmoothScrollProvider from "@/components/providers/SmoothScroll";
import "./globals.css";
import { CustomCursor } from "@/components/ui/CustomCursor";


/* ──────────────────────────────────────
   FONT LOADING — next/font/google
   Optimized subsets & weight ranges
────────────────────────────────────── */

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
  weight: ["300", "400", "500", "600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
  weight: ["400", "500", "600", "700"],
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
  weight: ["300", "400", "500", "600", "700"],
});

/* ──────────────────────────────────────
   METADATA
────────────────────────────────────── */

export const metadata: Metadata = {
  title: {
    template: "%s | NorthStar Energy Solutions",
    default: "NorthStar Energy Solutions | Simulation, Inspection & Chemicals",
  },
  description:
    "Integrated technical consultation, physical engineering, and advanced drilling simulation for the MENA and global energy sectors.",
  keywords: [
    "drilling simulation",
    "well control",
    "API 4G inspection",
    "NDT inspection",
    "industrial chemicals",
    "offshore marine services",
    "IWCF training",
    "IADC certification",
  ],
  metadataBase: new URL("https://northstarenergy.com"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://northstarenergy.com",
    title: "NorthStar Energy Solutions",
    description: "Integrated technical consultation, physical engineering, and advanced drilling simulation.",
    siteName: "NorthStar Energy Solutions",
  },
  twitter: {
    card: "summary_large_image",
    title: "NorthStar Energy Solutions",
    description: "Advanced drilling simulation and physical engineering for the energy sector.",
  },
  robots: {
    index: true,
    follow: true,
  }
};

/* ──────────────────────────────────────
   ROOT LAYOUT
────────────────────────────────────── */

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} ${spaceGrotesk.variable} font-sans bg-petroleum-900 text-white antialiased`}
      >
        <SmoothScrollProvider>
          <CustomCursor />
          {children}
        </SmoothScrollProvider>
      </body>
    </html>
  );
}
