import { Preloader } from "@/components/ui/Preloader";
import { EliteNavbar } from "@/components/ui/EliteNavbar";
import { IndustrialFooter } from "@/components/ui/IndustrialFooter";
import { InfiniteMarquee } from "@/components/ui/InfiniteMarquee";
import { SimulatorShowcase } from "@/components/ui/SimulatorShowcase";
import { CinematicShowcase } from "@/components/ui/CinematicShowcase";
import { ChemicalMatrix } from "@/components/ui/ChemicalMatrix";
import { ContactSection } from "@/components/ui/ContactSection";
import { HorizontalCapabilities } from "@/components/ui/HorizontalCapabilities";
import { FilmstripGallery } from "@/components/ui/FilmstripGallery";
import { ImpactMetrics } from "@/components/ui/ImpactMetrics";
import { HeroSection } from "@/components/ui/HeroSection";
import { NORTH_STAR_DATA } from "@/data/company";

export default function Home() {
  const { certifications, clients, services } = NORTH_STAR_DATA;

  return (
    <>
      <Preloader>
        <EliteNavbar />

        <main className="flex flex-col bg-void overflow-x-hidden">
          {/* 1. HERO */}
          <HeroSection />

          {/* 2. TRUST MARQUEE */}
          <section className="relative z-10 py-12 bg-void border-y border-white/[0.05]">
            <div className="mb-8 px-6 lg:px-12 text-center">
              <h3 className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/40">
                Global Accreditations
              </h3>
            </div>
            <InfiniteMarquee items={certifications} speed="slow" />
            
            <div className="mt-16 mb-8 px-6 lg:px-12 text-center">
              <h3 className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/40">
                Client Ecosystem — 38 Partners
              </h3>
            </div>
            <InfiniteMarquee items={clients} direction="right" speed="normal" />
          </section>

          {/* 3. CAPABILITIES */}
          <section id="capabilities" className="relative z-20">
            <HorizontalCapabilities />
          </section>

          {/* 4. RDS-100 SIMULATOR */}
          <section id="simulator" className="relative z-20">
            <SimulatorShowcase />
          </section>

          {/* 5. CINEMATIC SHOWCASE (Digital Twins) */}
          <section className="relative z-20">
            <CinematicShowcase />
          </section>

          {/* 6. CHEMICAL MATRIX */}
          <section id="chemicals" className="relative z-20">
            <ChemicalMatrix categories={services.chemicals.categories} />
          </section>

          {/* 7. FIELD OPS GALLERY */}
          <section id="field-ops" className="relative z-20 overflow-hidden">
            <FilmstripGallery />
          </section>

          {/* 8. IMPACT METRICS */}
          <section className="relative z-20">
            <ImpactMetrics />
          </section>

          {/* 9. CONTACT */}
          <section id="contact" className="relative z-20">
            <ContactSection />
          </section>
        </main>

        <IndustrialFooter />
      </Preloader>
    </>
  );
}
