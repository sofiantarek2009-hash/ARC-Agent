/**
 * DAD Company — Unified Data Layer
 * Single Brand Architecture
 * All structured company data consumed by the site renderer.
 */

window.DAD_DATA = {

  // ─────────────────────────────────────────────
  //  COMPANY PROFILE
  // ─────────────────────────────────────────────
  company: {
    name: 'DAD COMPANY',
    shortName: 'DAD',
    established: 2016,
    tagline: 'Pioneering Industrial Engineering & Advanced Drilling Simulation',
    profile: 'Reputable market leader well-known for extreme service quality, reliability, and high-tier customer care. We integrate the digital frontiers of drilling simulation with physical assurance and industrial engineering.',
    vision: 'To set the global benchmark for digitally enabled well operations and industrial inspection, where integrated simulation, engineering, and competency ecosystems redefine safety, performance, and execution certainty across the energy sector.',
    mission: 'Banking on our people as our most important asset, our mission is to provide reliable, innovative solutions that enable our customers to outperform safely. We pioneer the region\'s simulation technologies while delivering extreme-quality physical engineering assurance.',
    
    valueProps: [
      { title: 'Advanced Simulations', desc: 'Proprietary RDS-100® and WeXplore® platforms deliver globally approved, real-time enabled experiences' },
      { title: 'International Standards', desc: 'Absolute adherence to modern global quality standards ensures flawless consistency' },
      { title: 'Technical Expertise', desc: 'Exceptional technical history in heavy drilling operations backed by Shell, IWCF, and IADC' },
      { title: 'Value-for-Money', desc: 'Optimized supply chains combined with localized manufacturing allow competitive pricing' }
    ],

    // Hubs
    hubs: [
      { name: 'Service Hub', desc: 'Dedicated to safekeeping field manpower and capital infrastructure while helping clients excel at core business tracks' },
      { name: 'Sales Hub', desc: 'Manages B2B logistics and supply of state-of-the-art lifting, load-testing, and field equipment' },
      { name: 'Operation Hub', desc: 'Implements rigorous quality control measures to guarantee the safest field execution' }
    ]
  },

  // ─────────────────────────────────────────────
  //  SERVICE VERTICALS
  // ─────────────────────────────────────────────
  services: {
    
    // Technology & Simulation 
    technology: {
      id: 'technology',
      name: 'Technology & Simulation',
      subcategories: [
        {
          name: 'Simulators Manufacturing (RDS-100®)',
          desc: 'High-fidelity drilling and well control simulation built for real operations.',
          capabilities: [
            'Seamlessly integrates real-time operational data',
            'Continuous live data plotting',
            'Advanced stuck pipe and motor problem modules',
            'Well control scenario simulation'
          ]
        },
        {
          name: 'Virtual Reality (WEXPLORE®)',
          desc: 'Immersive, scenario-based environments exposing personnel to critical field risks.',
          capabilities: [
            '360-degree experience with full spatial awareness',
            'Direct ecosystem integration with RDS physical simulators',
            'Deeply oriented toward driving organizational HSE culture'
          ]
        },
        {
          name: 'Digital Solutions',
          desc: 'Driving workflow automation, predictive analytics, and real-time operational insights.',
          capabilities: [
            'Advanced workflow automation engines',
            'Intelligent well control and killing software',
            'Live tracking systems built directly for critical well operations',
            'Equipment PMS compliant with API and OEM specifications'
          ]
        }
      ]
    },

    // Engineering & Inspection
    engineering: {
      id: 'engineering',
      name: 'Engineering & Inspection',
      subcategories: [
        {
          name: 'Rig & Structural Inspection',
          desc: 'Certified inspections, engineering, and structural integrity assessments.',
          capabilities: [
            'Certified structural fall protection surveys',
            'Heavy Derrick Inspection per API 4G Cat III guidelines',
            'Drilling handling tools inspection and top drive inspection per API Cat III / IV',
            'High-altitude Rope Access services via IRATA-approved technicians'
          ]
        },
        {
          name: 'Offshore & Marine Services',
          desc: 'Comprehensive marine engineering and offshore structural inspection.',
          capabilities: [
            'Underwater Inspection in Lieu of Drydocking (UWILD)',
            'Precision ultrasonic hull gauging per ABS & DNVGL Class rules',
            'Complete marine NDT services (MPI/DPI/UT/EMI/VT/ECT)'
          ]
        },
        {
          name: 'Lifting Gear & Maintenance',
          desc: 'Equipment supply, maintenance, and heavy logistical support.',
          capabilities: [
            'Full inspection and certification of high-capacity lifting equipment (LGI)',
            'High-specification steel crane wire rope assets',
            'Top drive system engineering maintenance and OEM spare parts'
          ]
        },
        {
          name: 'Equipment Assessment',
          desc: 'Earthmoving and industrial equipment inspection.',
          capabilities: [
            'Industrial crane inspection by certified API RP 2D inspectors',
            'Industrial heavy forklift inspection regimes',
            'Fleet-wide commercial truck HSE safety assessments'
          ]
        }
      ]
    },

    // Training & Manpower
    humanCapital: {
      id: 'human-capital',
      name: 'Training & Competency',
      subcategories: [
        {
          name: 'Accredited Training',
          desc: 'Approved regional center certified by IWCF and IADC.',
          capabilities: [
            'Comprehensive well engineering fundamentals',
            'Hands-on drilling operations training with simulator integration',
            'Specialized well-intervention techniques'
          ]
        },
        {
          name: 'Competency Assurance',
          desc: 'Independent, evidence-based assessment of critical well operations roles.',
          capabilities: [
            'Written and Practical Examinations',
            'On-the-Job Assessment',
            'Technical Interviews',
            'High-stakes simulation-based assessments'
          ]
        },
        {
          name: 'Technical Manpower',
          desc: 'Provision of specialized field personnel.',
          capabilities: [
            'Lifting Coaches & Certified Lifting Inspectors',
            'Riggers & Qualified Welders',
            'Certified API Inspectors & HSE Engineers',
            'Industrial Divers & ALS Professionals'
          ]
        },
        {
          name: 'Corporate Consultation',
          desc: 'High-tier strategic engineering and management support.',
          capabilities: [
            'HSE Performance support systems',
            'Lean Six Sigma methodologies implementation',
            'Complex industrial capital project management',
            'Asset Integrity Management (AIM) solutions'
          ]
        }
      ]
    },

    // Industrial Chemicals
    chemicals: {
      id: 'chemicals',
      name: 'Industrial Chemicals',
      desc: 'Optimized economic efficiency across chemical treatment formulations, raw industrial elements, and active intermediaries.',
      categories: [
        {
          name: 'Drilling Fluids',
          products: ['Filter Cake Removal Systems', 'Fluid Loss Reducers', 'H2S Scavengers', 'Downhole Lubricants', 'Viscosifiers', 'Releasing Agents', 'Surfactants', 'H2S Control Systems']
        },
        {
          name: 'Commodity Elements',
          products: ['Calcium Chloride', 'Sodium Chloride', 'Calcium Oxide', 'Hydrochloric Acid (HCl)', 'Soda Ash', 'Premium Bentonite', 'Barite', 'Caustic Soda']
        },
        {
          name: 'Petrochemical Materials',
          products: ['High-purity Alcohols', 'Glycols', 'Ketones', 'Esters', 'Hydrocarbon Solvents']
        },
        {
          name: 'Specialty Treatments',
          products: ['Oil Dehydration (Demulsifiers)', 'Water Clarifiers', 'Foaming Products', 'Paraffin Control Agents', 'Asphaltene Control Compounds', 'Scale/Corrosion Inhibitors', 'Biocides']
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  //  COMBINED CERTIFICATIONS
  // ─────────────────────────────────────────────
  certifications: [
    { name: 'Shell', scope: 'Global drilling and well control simulation accreditation' },
    { name: 'IWCF', scope: 'International Well Control Forum — accredited training center certification' },
    { name: 'IADC', scope: 'International Association of Drilling Contractors — training approval' },
    { name: 'API-7K', scope: 'American Petroleum Institute — drilling equipment specification compliance' },
    { name: 'ISO', scope: 'International Organization for Standardization — quality management systems' },
    { name: 'ABS', scope: 'American Bureau of Shipping — marine and offshore structural certification' },
    { name: 'LEEA', scope: 'Lifting Equipment Engineers Association — lifting gear inspection standards' },
    { name: 'ASNT', scope: 'American Society for Nondestructive Testing — NDT personnel qualification' },
    { name: 'DNV-GL', scope: 'Det Norske Veritas Germanischer Lloyd — marine classification and assurance' },
    { name: 'AWS', scope: 'American Welding Society — welding procedure and personnel certification' },
    { name: 'NACE', scope: 'National Association of Corrosion Engineers — corrosion control standards' },
    { name: 'LIFT-TEX', scope: 'Textile lifting sling manufacturing and testing certification' }
  ],

  // ─────────────────────────────────────────────
  //  CLIENT ECOSYSTEM
  // ─────────────────────────────────────────────
  clients: [
    'BAPETCO', 'Enppi', 'Cheiron', 'Ryboa Haima (RH)', 'FOS Energy LLC', 'Halliburton', 'Nabors', 'Oxy',
    'Weatherford', 'Schlumberger', 'Petrogulf Misr', 'EDC', 'PMS', 'Suez Steel Co.',
    'UPC', 'EMC', 'Day', 'Petrobel', 'Saknafta', 'Saipem', 'Petro Masila', 'ADES',
    'Gempetco', 'Maridive Group', 'Kuwait Petroleum Corp.', 'Expro', 'Titan Naval Group',
    'Green Valley Oil Services', 'Sumed'
  ],

  // ─────────────────────────────────────────────
  //  GLOBAL PRESENCE
  // ─────────────────────────────────────────────
  contacts: {
    egypt: {
      city: 'Cairo',
      area: 'Nasr City / Maadi',
      details: 'Building 5027, Block 5, Meraj, Maadi, Cairo, Egypt',
      phones: ['+20-0220826046', '+20-1070787297', '+201212000531'],
      emails: ['info@dadcompany.com']
    },
    oman: {
      city: 'Muscat',
      area: 'Ghala Industrial Area',
      details: 'Ghala Industrial Area, Muscat, Oman',
      phones: [],
      emails: []
    },
    libya: {
      city: 'Tripoli',
      area: 'Tripoli',
      details: 'Tripoli, Libya',
      phones: [],
      emails: []
    }
  }
};
