import { EliteNavbar } from "@/components/ui/EliteNavbar";
import { IndustrialFooter } from "@/components/ui/IndustrialFooter";
import { ContactSection } from "@/components/ui/ContactSection";

export default function ContactPage() {
  return (
    <>
      <EliteNavbar />
      <main className="min-h-screen pt-24 bg-void">
        <ContactSection />
      </main>
      <IndustrialFooter />
    </>
  );
}
