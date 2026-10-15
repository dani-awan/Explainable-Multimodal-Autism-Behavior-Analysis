"""PDF download UI shared by existing research pages."""
def pdf_download(st,report,key):
    try:
        from report_pdf import build_pdf
        from research_prediction import prediction
        result=prediction(report)
        if result:
            st.subheader('ASD-related research prediction')
            st.write('**Model classification: '+result['classification']+'**')
            st.caption(result['scope'])
            st.info(result['interpretation'])
            st.write(result['next_step'])
            st.caption(result['evidence_limit'])
        payload=build_pdf(report)
        st.download_button('Download professional PDF report',payload,'research_analysis_report.pdf','application/pdf',key=key)
    except Exception as error:
        st.error('PDF export unavailable: '+str(error))
