"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { NORTH_STAR_DATA } from "@/data/company";

export function ContactSection() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    phone: "",
    message: "",
  });
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('loading');
    setErrorMessage("");

    try {
      const res = await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Failed to send message');
      }

      setStatus('success');
      setFormData({ name: "", email: "", company: "", phone: "", message: "" });
    } catch (err: any) {
      console.error('Contact form error:', err);
      setStatus('error');
      setErrorMessage(err.message || 'An unexpected error occurred. Please try again.');
    }
  };

  return (
    <section id="contact" className="w-full py-24 md:py-32 bg-[#0B0C0E] relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] pointer-events-none" />
      
      <div className="max-w-[1440px] mx-auto px-6 lg:px-12 relative z-10">
        <div className="flex flex-col lg:flex-row gap-16 lg:gap-24">
          
          <div className="w-full lg:w-1/2 flex flex-col justify-center">
            <div className="mb-12">
              <span className="font-mono text-xs text-hazard-orange uppercase tracking-widest block mb-4">
                Secure Transmission
              </span>
              <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6 tracking-tight">
                Initiate Project<br />Dialogue.
              </h2>
              <p className="font-sans text-sm text-white/50 max-w-xl">
                Connect with our corporate deployment teams or request an integrated consultation for your assets in the MENA region with NorthStar Energy Solutions.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="glass-panel p-6 rounded-lg border border-white/10 bg-[#14161D]/50 backdrop-blur-md">
                <h3 className="font-display font-bold text-white text-xl mb-2">Middle East HQ</h3>
                <p className="font-body text-white/70 mb-2">Maadi, Cairo, Egypt</p>
                <div className="flex items-center gap-3 text-white font-mono text-sm mb-2">
                  <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                  {NORTH_STAR_DATA.company.contact.hq.phones[0]}
                </div>
                <div className="flex items-center gap-3 text-white/60 font-mono text-xs">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  {NORTH_STAR_DATA.company.contact.hq.email}
                </div>
              </div>

              <div className="glass-panel p-6 rounded-lg border border-white/10 bg-[#14161D]/50 backdrop-blur-md">
                <h3 className="font-display font-bold text-white text-xl mb-2">Engineering & Inspection</h3>
                <p className="font-body text-white/70 mb-2">Cairo, Egypt</p>
                <div className="flex items-center gap-3 text-white font-mono text-sm mb-2">
                  <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                  {NORTH_STAR_DATA.company.contact.hq.phones[1]}
                </div>
                <div className="flex items-center gap-3 text-white/60 font-mono text-xs">
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  {NORTH_STAR_DATA.company.contact.aes.email}
                </div>
              </div>

              <div className="glass-panel p-6 rounded-lg border border-hazard-orange/30 bg-[#14161D]/50 backdrop-blur-md relative overflow-hidden sm:col-span-2">
                <div className="absolute top-0 right-0 p-6 flex items-center justify-center">
                  <div className="w-2 h-2 rounded-full bg-[#CCFF00] animate-pulse shadow-[0_0_8px_#CCFF00]" />
                </div>
                <h3 className="font-display font-bold text-white text-xl mb-2">24/7 Emergency</h3>
                <p className="font-body text-white/70 mb-0 flex items-center gap-2">
                  Available Around the Clock
                </p>
                <div className="flex items-center gap-3 text-white font-mono text-sm mt-4">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" /></svg>
                  {NORTH_STAR_DATA.company.contact.phone247}
                </div>
              </div>
            </div>
          </div>

          <div className="w-full lg:w-1/2">
            <div className="glass-panel-heavy p-8 rounded-xl border border-white/10 bg-[#0B0C0E]/80 backdrop-blur-xl">
              <form onSubmit={handleSubmit} className="flex flex-col gap-6">
                
                <div className="flex flex-col gap-2">
                  <label htmlFor="name" className="text-white/70 font-mono text-sm uppercase tracking-wide">Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="Enter your name"
                    className="w-full bg-[#14161D] border border-white/10 rounded-lg p-4 text-white font-mono text-sm placeholder:text-white/30 focus:outline-none focus:border-white/50 focus:ring-1 focus:ring-white/20 transition-colors"
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <label htmlFor="email" className="text-white/70 font-mono text-sm uppercase tracking-wide">Email</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    required
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="name@company.com"
                    className="w-full bg-[#14161D] border border-white/10 rounded-lg p-4 text-white font-mono text-sm placeholder:text-white/30 focus:outline-none focus:border-white/50 focus:ring-1 focus:ring-white/20 transition-colors"
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <label htmlFor="phone" className="text-white/70 font-mono text-sm uppercase tracking-wide">Phone Number</label>
                  <input
                    type="tel"
                    id="phone"
                    name="phone"
                    placeholder="+1 (555) 000-0000"
                    value={formData.phone}
                    onChange={handleChange}
                    className="w-full bg-[#14161D] border border-white/10 rounded-lg p-4 text-white font-mono text-sm placeholder:text-white/30 focus:outline-none focus:border-white/50 focus:ring-1 focus:ring-white/20 transition-colors"
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <label htmlFor="company" className="text-white/70 font-mono text-sm uppercase tracking-wide">Company</label>
                  <input
                    type="text"
                    id="company"
                    name="company"
                    required
                    value={formData.company}
                    onChange={handleChange}
                    placeholder="Your organization"
                    className="w-full bg-[#14161D] border border-white/10 rounded-lg p-4 text-white font-mono text-sm placeholder:text-white/30 focus:outline-none focus:border-white/50 focus:ring-1 focus:ring-white/20 transition-colors"
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <label htmlFor="message" className="text-white/70 font-mono text-sm uppercase tracking-wide">Message / RFQ Details</label>
                  <textarea
                    id="message"
                    name="message"
                    required
                    rows={4}
                    value={formData.message}
                    onChange={handleChange}
                    placeholder="How can we assist you? Include scope documents if necessary."
                    className="w-full bg-[#14161D] border border-white/10 rounded-lg p-4 text-white font-mono text-sm placeholder:text-white/30 focus:outline-none focus:border-white/50 focus:ring-1 focus:ring-white/20 transition-colors resize-y"
                  />
                </div>

                <div className="flex items-start gap-3 mt-2">
                  <input type="checkbox" id="consent" required className="mt-1 bg-[#14161D] border border-white/10 rounded text-hazard-orange focus:ring-hazard-orange" />
                  <label htmlFor="consent" className="text-white/50 text-xs font-sans leading-relaxed">
                    I consent to having NorthStar Energy Solutions process my personal data to handle my request. View our <a href="/privacy" className="text-hazard-orange hover:underline">Privacy Policy</a>.
                  </label>
                </div>

                {status === 'error' && (
                  <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 font-mono text-sm">
                    {errorMessage}
                  </div>
                )}

                {status === 'success' ? (
                  <div className="w-full bg-green-500/10 border border-green-500/20 text-green-400 font-bold font-mono uppercase tracking-wider py-4 rounded-lg flex items-center justify-center gap-3">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    MESSAGE RECEIVED
                  </div>
                ) : (
                  <button
                    type="submit"
                    disabled={status === 'loading'}
                    className="w-full bg-white text-[#0B0C0E] font-bold font-mono uppercase tracking-wider py-4 rounded-lg hover:bg-white/90 active:scale-[0.98] transition-all duration-200 flex items-center justify-center gap-3 disabled:opacity-70 disabled:cursor-not-allowed"
                  >
                    {status === 'loading' ? 'SUBMITTING...' : 'Request a Quote'}
                  </button>
                )}
              </form>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
