import { NextResponse } from 'next/server';
import { Resend } from 'resend';

// Initialize Resend with the API key from environment variables
// Fallback to a placeholder so it doesn't crash during build if not set
const resend = new Resend(process.env.RESEND_API_KEY || 're_placeholder');

interface ContactFormData {
  name: string;
  email: string;
  company: string;
  message: string;
}

export async function POST(request: Request) {
  try {
    // Basic rate limiting placeholder
    // TODO: Implement actual rate limiting using Redis/Vercel KV or similar
    
    const body: Partial<ContactFormData> = await request.json();
    const { name, email, company, message } = body;

    // Validation
    if (!name || !email || !company || !message) {
      return NextResponse.json(
        { error: 'All fields are required.' },
        { status: 400 }
      );
    }

    // Basic email validation regex
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: 'Invalid email address.' },
        { status: 400 }
      );
    }

    const recipient = process.env.CONTACT_EMAIL || 'info@well-experts.org';

    // Send email using Resend
    const { data, error } = await resend.emails.send({
      from: 'Well Experts Contact <contact@well-experts.org>', // Must be a verified domain in Resend
      to: [recipient],
      subject: `[Well Experts] New Inquiry from ${name} at ${company}`,
      html: `
        <h2>New Contact Form Submission</h2>
        <p><strong>Name:</strong> ${name}</p>
        <p><strong>Email:</strong> ${email}</p>
        <p><strong>Company:</strong> ${company}</p>
        <br/>
        <h3>Message:</h3>
        <p>${message.replace(/\n/g, '<br/>')}</p>
      `,
    });

    if (error) {
      console.error('Resend error:', error);
      return NextResponse.json(
        { error: 'Failed to send message. Please try again later.' },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { message: 'Message sent successfully.', id: data?.id },
      { status: 200 }
    );
  } catch (err) {
    console.error('Contact API error:', err);
    return NextResponse.json(
      { error: 'Internal server error.' },
      { status: 500 }
    );
  }
}
