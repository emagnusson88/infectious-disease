library(shiny)
library(coronavirus)
library(ggplot2)
library(dplyr)
library(readr)
library(reticulate)

# Get daily updates from GitHub
covid <- read.csv('https://raw.githubusercontent.com/RamiKrispin/coronavirus/master/csv/coronavirus.csv')

# format date
covid$date <- as.Date(covid$date, format =  "%Y-%m-%d")

# See above for the definitions of ui and server
ui <- fluidPage(
  titlePanel("Covid Forecasting"),
  
  sidebarLayout(
    sidebarPanel(
      selectizeInput("countryInput","Country",
                     choices=unique(covid$country),
                     selected="US", multiple=FALSE),
      radioButtons("dataSource","Data Source:", c("Upload Data","Use Pre-loaded Data")),
      uiOutput("ui"),
      actionButton("update","Update Chart")
    ),
    
    mainPanel(
      plotOutput("covidplot"), 
      h3('Python script output'),
      verbatimTextOutput('message')
    )
  )
)

server <- function(input,output){
  
  output$ui <- renderUI({
    if (is.null(input$dataSource))
      return()

    switch(input$dataSource,
           "Upload Data" = fileInput("dataFile","Choose CSV File",accept = ".csv")
    )
  })
  
  reticulate::source_python('python_test.py')
  
  output$message <- renderText({
    return(python_test())
  })
  
  
  c <- eventReactive(input$update,{
    covid %>%
        filter(country == input$countryInput,
               type == "confirmed",
               date >= '2021-06-01',
               date <= '2021-10-01')
  })
    
  output$covidplot <- renderPlot({
    ggplot(c(), aes(x=date,y=cases,color=country)) +
      geom_point() +
      geom_line() +
      xlab("Date") +
      ylab("Cases") +
      ggtitle("Cases Over Time")
  })
 
}

shinyApp(ui = ui, server = server)